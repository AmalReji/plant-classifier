from collections import Counter
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from torchvision.models import get_weight


def preprocess_images(dataset_dir: Path, model_name: str, batch_size: int = 32, num_workers: int = 4, sampling_method="none") -> DataLoader:
    """ Preprocess images in the dataset directory for a specific model.
    Args:
        dataset_dir (Path): Directory containing images organized in subfolders by class.
        model_name (str): Name of the model to use for preprocessing.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of worker threads for DataLoader.
        sampling_method (str): Method for balancing classes. Options: "over", "under", "none".
    Returns:
        DataLoader: PyTorch DataLoader with balanced classes.
    """

    # Get model-specific preprocessing
    weights = get_weight(f"{model_name}_Weights.DEFAULT")  # Returns the best available weights for the model
    preprocess = weights.transforms()  # Returns the transformations required for the chosen model

    dataset = datasets.ImageFolder(root=str(dataset_dir), transform=preprocess)

    # Count samples per class
    class_counts = Counter([class_index for image, class_index in dataset])
    sample_count = len(dataset)

    if sampling_method == "over":
        # Give minority classes a higher weight
        class_weights = {cls: sample_count / class_count for cls, class_count in class_counts.items()}
    elif sampling_method == "under":
        # Give majority classes a lower weight
        class_weights = {cls: 1.0 / class_count for cls, class_count in class_counts.items()}
    elif sampling_method == "none":
        # No balancing, all classes have equal weight
        class_weights = {cls: 1.0 for cls in class_counts.keys()}

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
