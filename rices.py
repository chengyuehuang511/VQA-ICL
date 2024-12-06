import open_clip
import torch
from tqdm import tqdm
import torch
import utils
import os
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP
from open_flamingo.eval.utils import unwrap_model

class RICES:
    def __init__(
        self,
        dataset,
        device,
        batch_size,
        vision_encoder_path="ViT-B-32",
        vision_encoder_pretrained="openai",
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
        vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
            vision_encoder_path,
            pretrained=vision_encoder_pretrained,
        )
        self.model = vision_encoder.to(self.device)
        self.model = DDP(self.model, device_ids=[self.device])
        self.image_processor = image_processor

        # Precompute features
        if os.path.exists(cached_features_path):
            self.features = torch.load(
                cached_features_path, map_location="cpu"
            )
        else:
            self.features = self._precompute_features(dataset)
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

    def _precompute_features(self, dataset):
        # Switch to evaluation mode
        self.model.eval()

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
                desc="Precomputing features for RICES",
            ):
                # print(batch["idx"])
                inputs = torch.stack(
                    [self.image_processor(image) for image in batch["image"]]
                ).to(self.device)
                image_features = unwrap_model(self.model).encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.cpu().detach()

                assert len(image_features) == len(batch["idx"])

                for feat_sample, sample_id in zip(image_features, batch["idx"]):
                    features_rank.append({"feature": feat_sample.unsqueeze(0), "idx": sample_id})

        # all gather
        features = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(features, features_rank)  # list of lists

        if self.rank != 0:
            return None

        # sort by idx: features is a list of jsons, each json has a feature and an idx
        features = sorted([item for sublist in features for item in sublist], key=lambda x: x["idx"])
        idx = [item["idx"] for item in features]
        features = [item["feature"].detach() for item in features]
        
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
        # Switch to evaluation mode
        self.model.eval()

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
