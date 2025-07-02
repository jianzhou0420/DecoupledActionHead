# First, make sure you have the required libraries installed.
# You can install them using pip:
# pip install transformers torch pillow requests

import torch
from PIL import Image
import requests
from transformers import Dinov2ImageProcessor, Dinov2Model


def get_dino_image_features(image_source):
    """
    Encodes an image using a pre-trained DINOv2 model.

    Args:
        image_source (str): URL or local file path of the image.

    Returns:
        torch.Tensor: The image features (embedding) as a PyTorch tensor.
    """
    # 1. Load the pre-trained DINOv2 model and its image processor.
    # We are using the base model. Other options include:
    # facebook/dinov2-small, facebook/dinov2-large, facebook/dinov2-giant
    model_name = "facebook/dinov2-base"

    # DINOv2 uses Dinov2ImageProcessor and Dinov2Model
    processor = Dinov2ImageProcessor.from_pretrained(model_name)
    model = Dinov2Model.from_pretrained(model_name)

    # Move the model to the appropriate device (GPU if available, otherwise CPU).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"Model loaded and moved to {device}.")

    # 2. Load and preprocess the image.
    try:
        if image_source.startswith('http://') or image_source.startswith('https://'):
            image = Image.open(requests.get(image_source, stream=True).raw).convert("RGB")
            print(f"Successfully downloaded image from URL: {image_source}")
        else:
            image = Image.open(image_source).convert("RGB")
            print(f"Successfully opened local image: {image_source}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    # Note: It's good practice to ensure the image is in RGB format.

    # 3. Process the image.
    # The processor handles resizing, normalization, and converting the image to a tensor.
    inputs = processor(images=image, return_tensors="pt")

    # Move the processed inputs to the same device as the model.
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print("Image preprocessed successfully.")

    # 4. Get the image features.
    # We use torch.no_grad() to disable gradient calculations for faster inference.
    with torch.no_grad():
        outputs = model(**inputs)

    # The DINOv2 model outputs a `pooler_output` which is a good representation
    # of the entire image.
    image_features = outputs.pooler_output

    print("Image features extracted.")
    return image_features


if __name__ == '__main__':
    # --- Example Usage ---

    # You can use a URL to an image...
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    # ...or a path to a local file.
    # For example: image_path = "path/to/your/image.jpg"

    # Get the embedding
    features = get_dino_image_features(image_url)

    if features is not None:
        print("\n--- Results ---")
        print("Shape of the image features tensor:", features.shape)
        # For the dinov2-base model, the embedding dimension is 768.
        print(f"This means we have 1 image with an embedding dimension of {features.shape[1]}.")
        print("\nImage Feature Vector (Embedding):")
        print(features)
