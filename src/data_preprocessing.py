from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.models import get_weight
from torch.utils.data import DataLoader
import cv2
from PIL import Image


def preprocess_images(dataset_dir: Path, model_name: str, img_size: int = 224):
    images = []
    labels = []

    # Warning: String is case-sensitive
    weights = get_weight(f"{model_name}_Weights.DEFAULT")  # Returns the best available weights for the model
    preprocess = weights.transforms()  # Returns the transformations required for the chosen model

    transforms_augmented = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomCrop(size=preprocess.crop_size[0], padding=4)
    ])

    for label in dataset_dir.iterdir():
        if label.is_dir():
            labels.append(label.name)
            for img_path in label.iterdir():
                # Read images
                original_dataset = datasets.ImageFolder(Path(dataset_dir,label), transform=preprocess)
                # Only augment enough images to reach max folder size
                augmented_dataset = datasets.ImageFolder(Path(dataset_dir, label), transform=transforms_augmented)
                train_dataset = torch.utils.data.ConcatDataset([original_dataset, augmented_dataset])
                images.append(train_dataset)
    return images, labels

    # img = Image.open(img_path)

    # Create transformation to mimic ResNet50 and MobileNetV2 preprocessing


    transform = transforms.Compose([
        transforms.resize((img_size, img_size)),
        transforms.ToTensor(),  # Also converts to [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Apply transformation to each image #


    #cwd = Path.cwd()
    #for label in dataset_dir.iterdir():
    #    if label.is_dir():
    #        labels.append(label.name)
    #        for img_path in label.iterdir():
    #            # Read images
    #            img = cv2.imread(str(img_path), )

    #            # Resize images
    #            img = cv2.resize(img, (img_size, img_size))

    #            # OPTIONAL: Change image color ordering from BGR to RGB
    #            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR





def get_dataloaders(data_dir: Path, batch_size: int = 32, img_size: int = 224):
    """
    Creates PyTorch DataLoaders for train, validation, and test sets from folder structure.

    Args:
        data_dir (Path): Path to the 'Plants_2' directory containing 'train', 'valid', 'test'.
        batch_size (int): Number of samples per batch.
        img_size (int): Resize images to img_size x img_size.

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    # Define transforms
    common_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Assumes grayscale-like image, change for RGB if needed
    ])

    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"
    test_dir = data_dir / "test"

    # Create datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=common_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=common_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=common_transforms)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    root_dir = Path.joinpath(Path.cwd(), "..")
    data_path = Path.joinpath(root_dir, "data", "Plants_2")
    train_dir = Path.joinpath(data_path,"train")
    valid_dir = Path.joinpath(data_path, "valid")
    test_dir = Path.joinpath(data_path,"test")

    #for label in test_dir.glob("*/"):
    #    for img_path in label.glob("*.jpg"):
    #        # Read images
    #        img = cv2.imread(str(img_path))

    #cv2.imshow(img_path.name, img)

    preprocess_images(Path(valid_dir), model_name="ResNet50", img_size=224)

    #train_loader, valid_loader, test_loader = get_dataloaders(data_path)

    #print(f"Loaded {len(train_loader.dataset)} training images.")
    #print(f"Loaded {len(valid_loader.dataset)} validation images.")
    #print(f"Loaded {len(test_loader.dataset)} test images.")
