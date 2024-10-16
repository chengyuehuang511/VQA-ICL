import open_clip
import torch
from tqdm import tqdm
import torch
from utils import custom_collate_fn


class JICES:
    def __init__(
        self,
        dataset,
        device,
        batch_size,
        eval_model,
        cached_features=None,
    ):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size

        # Load the model and processor
        self.model = eval_model

        # Precompute features
        if cached_features is None:
            self.features = self._precompute_features()
        else:
            self.features = cached_features

    def _precompute_features(self):
        features = []

        # Set up loader
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=custom_collate_fn,
        )

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
                    batch_text.append(
                        self.model.get_vqa_prompt(question=batch["question"][i])
                    )
                image_tensor = self.model._prepare_images(batch_images)
                text_tensor, attention_mask = self.model._prepare_text(batch_text)

                joint_features = self.model.__call__(
                    lang_x=text_tensor,
                    vision_x=image_tensor,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                ).hidden_states[-1][:, -1, :].clone()
                joint_features /= joint_features.norm(dim=-1, keepdim=True)
                features.append(joint_features.detach())

        features = torch.cat(features)
        return features

    def find(self, batch, num_examples):
        """
        Get the top num_examples most similar examples to the images.
        """

        with torch.no_grad():
            batch_images, batch_text = [], []
            for i in range(len(batch["image"])):
                batch_images.append([batch["image"][i]])
                batch_text.append(
                    self.model.get_vqa_prompt(question=batch["question"][i])
                )
            image_tensor = self.model._prepare_images(batch_images)
            text_tensor, attention_mask = self.model._prepare_text(batch_text)

            # Get the feature of the input image
            query_feature = self.model.__call__(
                lang_x=text_tensor,
                vision_x=image_tensor,
                attention_mask=attention_mask,
                output_hidden_states=True,
            ).hidden_states[-1][:, -1, :].clone()
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
            query_feature = query_feature.detach()

            if query_feature.ndim == 1:
                query_feature = query_feature.unsqueeze(0)

            # Compute the similarity of the input image to the precomputed features
            similarity = (query_feature @ self.features.T).squeeze()

            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)

            # Get the indices of the 'num_examples' most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :num_examples]

        # Return with the most similar images last
        return [[self.dataset[i] for i in reversed(row)] for row in indices]
