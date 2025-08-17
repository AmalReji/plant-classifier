import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import get_model
from data_preprocessing import preprocess_images


def extract_features(dataloader: DataLoader, model_name: str, device: str = 'cpu') -> tuple:
    """ Extract features from images using a pre-trained model.
    Args:
        dataloader (DataLoader): DataLoader containing images.
        model_name (str): Name of the pre-trained model.
        device (str): Device to run the model on ('cpu' or 'cuda').
    Returns:
        tuple: Numpy arrays of extracted features and corresponding labels.
    """
    all_features = []
    all_labels = []

    # Use pre-trained model without final classification layer
    model = get_model(model_name, weights='DEFAULT')  # Use 'DEFAULT' to get the best available weights
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the final FC layer
    model.eval()  # Set to inference mode

    model.to(device)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = model(images)
            features = features.view(features.size(0), -1)  # Flatten (B, 2048, 1, 1) -> (B, 2048)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return (
        np.concatenate(all_features, axis=0),
        np.concatenate(all_labels, axis=0)
    )

if __name__ == "__main__":
    root_dir = Path.joinpath(Path.cwd(), "..")
    data_path = Path.joinpath(root_dir, "data", "Plants_2")
    train_dir = Path.joinpath(data_path, "train")
    valid_dir = Path.joinpath(data_path, "valid")
    test_dir = Path.joinpath(data_path, "test")

    # Read hyperparameters from JSON
    hyperparameters_file = Path(f"model_hyperparameters.json")
    if hyperparameters_file.exists():
        with open(hyperparameters_file, 'r') as f:
            hyperparameters = json.load(f)
    sampling_method = hyperparameters["sampling_method"]
    num_workers = hyperparameters["num_workers"]
    model_name = hyperparameters["model_name"]
    batch_size = hyperparameters["batch_size"]

    # Preprocess images and create DataLoaders
    # start_time = time.time()
    # train_loader = preprocess_images(dataset_dir=Path(valid_dir), model_name=model_name, batch_size=batch_size,
    #                                      num_workers=num_workers, sampling_method=sampling_method)
    # end_time = time.time()
    # print(f"Loaded {len(train_loader.dataset)} training images in {end_time - start_time} secs.")

    start_time = time.time()
    valid_loader = preprocess_images(dataset_dir=Path(valid_dir), model_name=model_name, batch_size=batch_size,
                                     num_workers=num_workers, sampling_method=sampling_method)
    end_time = time.time()
    print(f"Loaded {len(valid_loader.dataset)} validation images in {end_time - start_time} secs.")

    start_time = time.time()
    test_loader = preprocess_images(dataset_dir=Path(valid_dir), model_name=model_name, batch_size=batch_size,
                                     num_workers=num_workers, sampling_method=sampling_method)
    end_time = time.time()
    print(f"Loaded {len(test_loader.dataset)} test images in {end_time - start_time} secs.")

    # Extract features from the datasets
    # start_time = time.time()
    # train_X, train_y = extract_features(dataloader=train_loader, model_name=model_name)
    # end_time = time.time()
    # print(f"Extracted features from {len(train_y)} training images in {end_time - start_time} secs.")

    start_time = time.time()
    valid_X, valid_y = extract_features(dataloader=valid_loader, model_name=model_name)
    end_time = time.time()
    print(f"Extracted features from {len(valid_y)} validation images in {end_time - start_time} secs.")

    start_time = time.time()
    test_X, test_y = extract_features(dataloader=test_loader, model_name=model_name)
    end_time = time.time()
    print(f"Extracted features from {len(test_y)} test images in {end_time - start_time} secs.")

    # Save the extracted features and labels to .npy files
    # np.save(file='train_X.npy', arr=train_X)
    # np.save(file='train_y.npy', arr=train_y)
    np.save(file='valid_X.npy', arr=valid_X)
    np.save(file='valid_y.npy', arr=valid_y)
    np.save(file='test_X.npy', arr=test_X)
    np.save(file='test_y.npy', arr=test_y)