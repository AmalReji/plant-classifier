import argparse
from pathlib import Path
import numpy as np
from data_preprocessing import preprocess_images
from extract_features import extract_features

'''This script extracts features from images in a dataset for hyperparameter tuning.'''

def main():
    parser = argparse.ArgumentParser(description='Extract features for hyperparameter tuning.')
    parser.add_argument('--model_name', type=str, required=True, help='Model to use for feature extraction.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for DataLoader.')
    parser.add_argument('--sampling_method', type=str, default='none', choices=['over', 'under', 'none'],
                        help='Method for balancing classes: "oversampling", "undersampling", or "none".')

    args = parser.parse_args()

    # Set up paths
    root_dir = Path.cwd().parent
    data_path = Path.joinpath(root_dir, "data", "Plants_2")
    train_dir = Path.joinpath(data_path, "train")
    valid_dir = Path.joinpath(data_path, "valid")
    test_dir = Path.joinpath(data_path, "test")

    print(f"Using model: {args.model_name}")
    print(f"Batch size: {args.batch_size}, Num workers: {args.num_workers}, Sampling method: {args.sampling_method}")

    try:
        # Preprocess images and create DataLoaders
        train_loader = preprocess_images(
            dataset_dir=train_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampling_method=args.sampling_method
        )
        print(f"Loaded {len(train_loader.dataset)} training images.")

        valid_loader = preprocess_images(
            dataset_dir=valid_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampling_method='none'  # No balancing for validation set
        )
        print(f"Loaded {len(valid_loader.dataset)} validation images.")

        test_loader = preprocess_images(
            dataset_dir=test_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampling_method='none'  # No balancing for test set
        )
        print(f"Loaded {len(test_loader.dataset)} test images.")

        # Extract features from the datasets
        train_X, train_y = extract_features(dataloader=train_loader, model_name=args.model_name)
        print(f"Extracted features from {len(train_y)} training images.")
        valid_X, valid_y = extract_features(dataloader=valid_loader, model_name=args.model_name)
        print(f"Extracted features from {len(valid_y)} validation images.")
        test_X, test_y = extract_features(dataloader=test_loader, model_name=args.model_name)
        print(f"Extracted features from {len(test_y)} test images.")

        # Save the extracted features and labels
        np.save('train_X.npy', train_X)
        np.save('train_y.npy', train_y)
        np.save('valid_X.npy', valid_X)
        np.save('valid_y.npy', valid_y)
        np.save('test_X.npy', test_X)
        np.save('test_y.npy', test_y)

        print("Feature extraction completed and saved to .npy files.")

    except Exception as e:
        print(f"Error during feature extraction: {e}")
        raise

if __name__ == "__main__":
    main()
