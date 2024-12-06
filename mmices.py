import open_clip
import torch
from tqdm import tqdm
import utils
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP
from open_flamingo.eval.utils import unwrap_model

class MMICES:
    def __init__(
        self,
        dataset,
        device,
        batch_size,
        vision_encoder_path="ViT-B-32",
        vision_encoder_pretrained="openai",
        lm_path="anas-awadalla/mpt-1b-redpajama-200b",
        lm_tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
        cached_features_path=None,
        query_dataset=None,
        query_cached_features_path=None,
    ):
        self.dataset = dataset
        self.query_dataset = query_dataset
        self.device = device
        self.batch_size = batch_size

        self.query_cached_features_path = query_cached_features_path
        self.cached_features_path = cached_features_path

        self.local_rank, self.rank, self.world_size = world_info_from_env()

        # Load the vision model and processor
        vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
            vision_encoder_path,
            pretrained=vision_encoder_pretrained,
        )
        self.model = vision_encoder.to(self.device)
        self.image_processor = image_processor

        # Load the language model and tokenizer
        lang_encoder = AutoModelForCausalLM.from_pretrained(
            lm_path,
            trust_remote_code=True,
        )
        text_tokenizer = AutoTokenizer.from_pretrained(
            lm_tokenizer_path,
            trust_remote_code=True,
        )
        # add Flamingo special tokens to the tokenizer
        text_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
        )
        if text_tokenizer.pad_token is None:
            # Issue: GPT models don't have a pad token, which we use to
            # modify labels for the loss.
            text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.language_model = lang_encoder.to(self.device)
        self.tokenizer = text_tokenizer

        # Precompute features
        if os.path.exists(cached_features_path):
            self.features = torch.load(
                cached_features_path, map_location="cpu"
            )
            self.lang_features = self.features["lang_features"]
            self.features = self.features["features"]
        else:
            self.features, self.lang_features = self._precompute_features(dataset)
            if self.rank == 0:
                os.makedirs(os.path.dirname(cached_features_path), exist_ok=True)
                torch.save(
                    {"features": self.features, "lang_features": self.lang_features}, 
                    cached_features_path)
            # Synchronize all processes
            torch.distributed.barrier()
            # Now all ranks load the features
            if self.rank != 0:
                self.features = torch.load(cached_features_path, map_location="cpu")["features"]
                self.lang_features = torch.load(cached_features_path, map_location="cpu")["lang_features"]

        if os.path.exists(query_cached_features_path):
            self.query_features = torch.load(
                query_cached_features_path, map_location="cpu"
            )
            self.query_lang_features = self.query_features["lang_features"]
            self.query_features = self.query_features["features"]
        else:
            self.query_features, self.query_lang_features = self._precompute_features(query_dataset)
            if self.rank == 0:
                os.makedirs(os.path.dirname(query_cached_features_path), exist_ok=True)
                torch.save(
                    {"features": self.query_features, "lang_features": self.query_lang_features}, 
                    query_cached_features_path)
            torch.distributed.barrier()
            if self.rank != 0:
                self.query_features = torch.load(query_cached_features_path, map_location="cpu")["features"]
                self.query_lang_features = torch.load(query_cached_features_path, map_location="cpu")["lang_features"]

        self.features = self.features.to(self.device)
        self.lang_features = self.lang_features.to(self.device)
        self.query_features = self.query_features.to(self.device)
        self.query_lang_features = self.query_lang_features.to(self.device)

        assert len(self.features) == len(dataset)
        assert len(self.query_features) == len(query_dataset)

    def _precompute_features(self, dataset):
        # Switch to evaluation mode
        self.model.eval()
        self.language_model.eval()

        # Set up loader
        loader = utils.prepare_eval_samples(
            dataset,
            len(dataset),
            self.batch_size,
        )

        features_rank = []
        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc="Precomputing features for MMICES",
            ):
                # Precompute image features
                inputs = torch.stack(
                    [self.image_processor(image) for image in batch["image"]]
                ).to(self.device)
                image_features = self.model.encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.cpu().detach()

                # Precompute language features
                text = self.tokenizer(batch["question"], padding=True, return_tensors="pt").to(self.device)
                lang_features_sample = self.language_model(**text, output_hidden_states=True).hidden_states[-1][:, -1, :]
                lang_features_sample /= lang_features_sample.norm(dim=-1, keepdim=True)
                lang_features_sample = lang_features_sample.cpu().detach()

                assert len(image_features) == len(batch["idx"])
                assert len(lang_features_sample) == len(batch["idx"])

                for feat, lang_feat, sample_id in zip(image_features, lang_features_sample, batch["idx"]):
                    features_rank.append({"image_feature": feat.unsqueeze(0), "lang_feature": lang_feat.unsqueeze(0), "idx": sample_id})

        # all gather
        features = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(features, features_rank)  # list of lists

        if self.rank != 0:
            return None, None
        
        # sort by idx: features is a list of jsons, each json has a feature and an idx
        features = sorted([item for sublist in features for item in sublist], key=lambda x: x["idx"])
        idx = [item["idx"] for item in features]
        lang_features = [item["lang_feature"].detach() for item in features]
        features = [item["image_feature"].detach() for item in features]

        # remove duplicates in idx and corresponding features
        # Initialize an empty set to track seen indices
        seen = set()
        # Initialize empty lists to store unique indices and corresponding features
        unique_idx = []
        unique_features = []
        unique_lang_features = []

        # Iterate over the idx and features lists simultaneously
        for i, feature, lang_feature in zip(idx, features, lang_features):
            if i not in seen:
                # If the index hasn't been seen, add it to the set and append the index and feature to the unique lists
                seen.add(i)
                unique_idx.append(i)
                unique_features.append(feature)
                unique_lang_features.append(lang_feature)

        # Update the original lists to the unique lists
        idx = unique_idx
        features = unique_features
        lang_features = unique_lang_features

        features = torch.cat(features)
        lang_features = torch.cat(lang_features)
        return features, lang_features

    def find(self, batch, num_examples, K=200):
        """
        Get the top num_examples most similar examples to the images.
        """
        # Switch to evaluation mode
        self.model.eval()
        self.language_model.eval()

        with torch.no_grad():
            """Choose the top K images"""
            query_feature = self.query_features[batch["idx"]]

            # Compute the similarity of the input image to the precomputed features
            similarity = (query_feature @ self.features.T).squeeze()

            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)

            # Get the indices of the K most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :K]

            """Among the top K images, choose the top num_examples texts"""
            query_lang_features = self.query_lang_features[batch["idx"]]
            
            # Initialize a list to store the selected top text indices
            selected_text_indices = []

            # Iterate over the batch to compute similarity for top K images and select top texts
            for i in range(query_feature.shape[0]):
                top_k_indices = indices[i]
                top_k_text_features = self.lang_features[top_k_indices]
                
                # Compute similarity between language features and the top K image features
                similarity_text = (query_lang_features[i] @ top_k_text_features.T).squeeze()
                
                # Get the indices of the top num_examples most similar texts for the current image
                top_text_indices = similarity_text.argsort(descending=True)[:num_examples]
                selected_text_indices.append(top_text_indices)
            
            """
            # Compute the similarity of the input text to the precomputed features
            similarity_text = (lang_features @ self.lang_features.T).squeeze()

            if similarity_text.ndim == 1:
                similarity_text = similarity_text.unsqueeze(0)
            
            # Get the indices of the 'num_examples' most similar texts
            indices_text = similarity_text.argsort(dim=-1, descending=True)[:, :num_examples]
            """

        # Return with the most similar images last
        # print("indices", indices)
        # print("selected_text_indices", selected_text_indices)
        return [[self.dataset[i] for i in reversed(row[selected_text_indices[row_id]])] for row_id, row in enumerate(indices)]

