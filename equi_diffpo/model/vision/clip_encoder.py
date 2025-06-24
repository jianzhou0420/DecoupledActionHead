# First, make sure you have the required libraries installed.
# You can install them using pip:
# pip install transformers torch pillow requests safetensors torchvision

import torch
import torchvision.transforms as T
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from typing import Union, List

import torch.nn as nn


class CLIPImageEncoder(nn.Module):

    def __init__(self):
        super(CLIPImageEncoder, self).__init__()

        model_name = "openai/clip-vit-base-patch32"
        # Load the pre-trained CLIP model and processor
        try:
            self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
        except Exception as e:
            self.model = CLIPModel.from_pretrained(model_name)

        self.processor = CLIPProcessor.from_pretrained(model_name)

        # --- Create separate preprocessing pipelines for train and eval ---
        image_processor = self.processor.image_processor
        resize_transform = T.Resize((image_processor.crop_size['height'], image_processor.crop_size['width']), antialias=True)
        normalize_transform = T.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

        # Preprocessor for training mode (uses random cropping)
        self.train_preprocessor = T.Compose([
            T.RandomCrop((76, 76)),
            resize_transform,
            normalize_transform,
        ])

        # Preprocessor for evaluation mode (uses center cropping)
        self.eval_preprocessor = T.Compose([
            T.CenterCrop((76, 76)),
            resize_transform,
            normalize_transform,
        ])

        # required grad false
        for param in self.model.parameters():
            param.requires_grad = False

    def _preprocess_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:

        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

        if self.training:
            # Use the training preprocessor (RandomCrop)
            processed_tensor = self.train_preprocessor(image_tensor)
        else:
            # Use the evaluation preprocessor (CenterCrop)
            processed_tensor = self.eval_preprocessor(image_tensor)

        return processed_tensor

    def encode(self, image_source: Union[str, List[str], Image.Image, List[Image.Image], torch.Tensor]) -> torch.Tensor:

        pixel_values = None

        if isinstance(image_source, torch.Tensor):
            pixel_values = self._preprocess_tensor(image_source)
        else:
            raise NotImplementedError(
                "Currently, only torch.Tensor inputs are supported. "
                "Please provide a tensor of shape(B, C, H, W) with pixel"
                " values in [0, 255] or [0, 1]."
            )

        with torch.no_grad():
            image_features = self.model.get_image_features(pixel_values=pixel_values)

        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        return image_features


if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Initialize the encoder. It starts in eval mode by default.
    encoder = CLIPImageEncoder()

    # 2. Create a sample tensor
    my_tensor = torch.randint(0, 256, (1, 3, 86, 86), dtype=torch.float32)
    print(f"\nCreated a sample tensor with shape: {my_tensor.shape}")

    # 3. Encode in EVALUATION mode (default)
    print("\n--- Running in EVAL mode (CenterCrop) ---")
    eval_features = encoder.encode(my_tensor)
    if eval_features is not None:
        print(f"Shape of eval features tensor: {eval_features.shape}")

    # 4. Switch to TRAINING mode
    print("\n--- Switching to TRAIN mode (RandomCrop) ---")
    encoder.train()

    # 5. Encode in TRAINING mode
    # Note: The result will be different each time due to RandomCrop
    train_features_1 = encoder.encode(my_tensor)
    train_features_2 = encoder.encode(my_tensor)
    if train_features_1 is not None:
        print(f"Shape of train features tensor: {train_features_1.shape}")
        # Check if the random crops produce different results
        are_different = not torch.allclose(train_features_1, train_features_2)
        print(f"Two runs in train mode produced different results: {are_different}")

    # 6. Switch back to EVALUATION mode
    print("\n--- Switching back to EVAL mode (CenterCrop) ---")
    encoder.eval()
    eval_features_1 = encoder.encode(my_tensor)
    eval_features_2 = encoder.encode(my_tensor)
    if eval_features_1 is not None:
        # Check if center crops produce the same results
        are_same = torch.allclose(eval_features_1, eval_features_2)
        print(f"Two runs in eval mode produced the same results: {are_same}")
