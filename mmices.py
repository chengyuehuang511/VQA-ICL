import open_clip
import torch
from tqdm import tqdm
import torch
from utils import custom_collate_fn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


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
    ):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size

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
            self.features, self.lang_features = self._precompute_features()
            torch.save(
                {"features": self.features, "lang_features": self.lang_features}, 
                cached_features_path)

    def _precompute_features(self):
        features = []
        lang_features = []

        # Switch to evaluation mode
        self.model.eval()
        self.language_model.eval()

        # Set up loader
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=custom_collate_fn,
        )

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
                features.append(image_features.detach())

                # Precompute language features
                text = self.tokenizer(batch["question"], padding=True, return_tensors="pt").to(self.device)
                lang_features_sample = self.language_model(**text, output_hidden_states=True).hidden_states[-1][:, -1, :]
                lang_features_sample /= lang_features_sample.norm(dim=-1, keepdim=True)
                lang_features.append(lang_features_sample)

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
            inputs = torch.stack([self.image_processor(image) for image in batch["image"]]).to(
                self.device
            )

            # Get the feature of the input image
            query_feature = self.model.encode_image(inputs)
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
            query_feature = query_feature.detach()

            if query_feature.ndim == 1:
                query_feature = query_feature.unsqueeze(0)

            # Compute the similarity of the input image to the precomputed features
            similarity = (query_feature @ self.features.T).squeeze()

            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)

            # Get the indices of the K most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :K]

            """Among the top K images, choose the top num_examples texts"""
            text = self.tokenizer(batch["question"], padding=True, return_tensors="pt").to(self.device)
            query_lang_features = self.language_model(**text, output_hidden_states=True).hidden_states[-1][:, -1, :]
            query_lang_features /= query_lang_features.norm(dim=-1, keepdim=True)
            query_lang_features = query_lang_features.detach()

            if query_lang_features.ndim == 1:
                query_lang_features = query_lang_features.unsqueeze(0)
            
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

