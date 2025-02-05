import open_clip
import torch
from tqdm import tqdm
import torch
import utils
import os
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP
from open_flamingo.eval.utils import unwrap_model
import numpy as np

class JICES:
    def __init__(
        self,
        dataset,
        device,
        batch_size,
        eval_model,
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

        # Load the model and processor
        self.model = eval_model

        # Precompute features
        if os.path.exists(cached_features_path):
            self.features = torch.load(
                cached_features_path, map_location="cpu"
            )
        else:
            self.features = self._precompute_features(dataset, is_train=False)
            if self.rank == 0:
                os.makedirs(os.path.dirname(cached_features_path), exist_ok=True)
                torch.save(self.features, cached_features_path)
            # Synchronize all processes
            torch.distributed.barrier()
            # Now all ranks load the features
            if self.rank != 0:
                self.features = torch.load(cached_features_path, map_location="cpu")
        
        if os.path.exists(query_cached_features_path):
            self.query_features = torch.load(
                query_cached_features_path, map_location="cpu"
            )
        else:
            self.query_features = self._precompute_features(query_dataset)
            if self.rank == 0:
                os.makedirs(os.path.dirname(query_cached_features_path), exist_ok=True)
                torch.save(self.query_features, query_cached_features_path)
            torch.distributed.barrier()
            if self.rank != 0:
                self.query_features = torch.load(query_cached_features_path, map_location="cpu")
        
        self.features = self.features.to(self.device)
        self.query_features = self.query_features.to(self.device)

        assert len(self.features) == len(dataset)
        assert len(self.query_features) == len(query_dataset)

    def _precompute_features(self, dataset, is_train=False):
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
                desc="Precomputing features for JICES",
            ):
                # question + answer
                # batch_text = [self.model.get_vqa_prompt(question=batch["question"][i], answer=max(batch["answers"][i], key=batch["answers"][i].count)) for i in range(len(batch["question"]))]
                
                # Convert the images and text to tensors
                batch_images, batch_text = [], []
                for i in range(len(batch["image"])):
                    batch_images.append([batch["image"][i]])
                    if is_train:
                        batch_text.append(
                            self.model.get_vqa_prompt(question=batch["question"][i], answer=batch["answers"][i][0])
                        )
                    else:
                        batch_text.append(
                            self.model.get_vqa_prompt(question=batch["question"][i])
                        )
                image_tensor = self.model._prepare_images(batch_images)
                text_tensor, attention_mask = self.model._prepare_text(batch_text)

                # joint_features = self.model.__call__(
                #     lang_x=text_tensor,
                #     vision_x=image_tensor,
                #     attention_mask=attention_mask,
                #     output_hidden_states=True,
                # ).hidden_states[-1][:, -1, :].clone()

                # joint_features = self.model(
                #     lang_x=text_tensor,
                #     vision_x=image_tensor,
                #     attention_mask=attention_mask,
                #     output_hidden_states=True,
                # ).hidden_states[-1].mean(dim=1)  # TODO: mean or last, mask out question padded tokens

                hidden_states = self.model(
                    lang_x=text_tensor,
                    vision_x=image_tensor,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                ).hidden_states
                joint_features = torch.sum(hidden_states[-1] * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask.unsqueeze(-1), dim=1) 

                joint_features /= joint_features.norm(dim=-1, keepdim=True)
                joint_features = joint_features.cpu().detach().numpy()  # For NCCL-based processed groups, internal tensor representations of objects must be moved to the GPU device before communication takes place.
                
                assert len(joint_features) == len(batch["idx"])

                for feat_sample, sample_id in zip(joint_features, batch["idx"]):
                    features_rank.append({"feature": np.expand_dims(feat_sample, axis=0), "idx": sample_id})

        # all gather
        features = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(features, features_rank)  # list of lists

        if self.rank != 0:
            return None

        # sort by idx: features is a list of jsons, each json has a feature and an idx
        features = sorted([item for sublist in features for item in sublist], key=lambda x: x["idx"])
        idx = [item["idx"] for item in features]
        features = [torch.from_numpy(item["feature"]) for item in features]
        
        # remove duplicates in idx and corresponding features
        # Initialize an empty set to track seen indices
        seen = set()
        # Initialize empty lists to store unique indices and corresponding features
        unique_idx = []
        unique_features = []

        # Iterate over the idx and features lists simultaneously
        for i, feature in zip(idx, features):
            if i not in seen:
                # If the index hasn't been seen, add it to the set and append the index and feature to the unique lists
                seen.add(i)
                unique_idx.append(i)
                unique_features.append(feature)

        # Update the original lists to the unique lists
        idx = unique_idx
        features = unique_features
        
        features = torch.cat(features)
        # print("idx", idx)
        # print("features.shape", features.shape)
        return features

    def find(self, batch, num_examples):
        """
        Get the top num_examples most similar examples to the images.
        """

        with torch.no_grad():
            query_feature = self.query_features[batch["idx"]]

            # Compute the similarity of the input image to the precomputed features
            similarity = (query_feature @ self.features.T).squeeze()

            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)

            # Get the indices of the 'num_examples' most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :num_examples]

        # Return with the most similar images last
        return [[self.dataset[i] for i in reversed(row)] for row in indices]
