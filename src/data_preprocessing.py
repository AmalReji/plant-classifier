import time
from collections import Counter
from pathlib import Path

import PIL
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.models import get_weight
from torch.utils.data import DataLoader, WeightedRandomSampler
import cv2
from PIL import Image


def preprocess_images(dataset_dir: Path, model_name: str, batch_size: int = 32, num_workers: int = 4) -> DataLoader:
    """ Preprocess images in the dataset directory for a specific model.
    Args:
        dataset_dir (Path): Directory containing images organized in subfolders by class.
        model_name (str): Name of the model to use for preprocessing.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of worker threads for DataLoader.
    Returns:
        DataLoader: PyTorch DataLoader with balanced classes.
    """

    # Get model-specific preprocessing
    weights = get_weight(f"{model_name}_Weights.DEFAULT")  # Returns the best available weights for the model
    preprocess = weights.transforms()  # Returns the transformations required for the chosen model

    dataset = datasets.ImageFolder(root=str(dataset_dir), transform=preprocess)

    # Count samples per class
    class_counts = Counter([class_index for image, class_index in dataset])

    # Give minority classes a higher weight
    sample_count = len(dataset)
    class_weights = {cls: sample_count / class_count for cls, class_count in class_counts.items()}

    # Compute weights for each image
    image_weights = [class_weights[class_index] for image, class_index in dataset]

    # Balance classes using a sampler
    sampler = WeightedRandomSampler(weights=image_weights, num_samples=sample_count, replacement=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    return dataloader


# if __name__ == "__main__":
#     root_dir = Path.joinpath(Path.cwd(), "..")
#     data_path = Path.joinpath(root_dir, "data", "Plants_2")
#     train_dir = Path.joinpath(data_path, "train")
#     valid_dir = Path.joinpath(data_path, "valid")
#     test_dir = Path.joinpath(data_path, "test")
#
#     model_name = "ResNet50"
#
#     start_time = time.time()
#     train_loader = preprocess_images(Path(train_dir), model_name=model_name)
#     end_time = time.time()
#     print(f"Loaded {len(train_loader.dataset)} training images in {end_time - start_time} secs.")
#
#     start_time = time.time()
#     valid_loader = preprocess_images(Path(valid_dir), model_name=model_name)
#     end_time = time.time()
#     print(f"Loaded {len(valid_loader.dataset)} validation images in {end_time - start_time} secs.")
#
#     start_time = time.time()
#     test_loader = preprocess_images(Path(test_dir), model_name=model_name)
#     end_time = time.time()
#     print(f"Loaded {len(test_loader.dataset)} test images in {end_time - start_time} secs.")
