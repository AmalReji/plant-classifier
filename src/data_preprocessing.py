from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2


def preprocess_images(dataset_dir: Path, img_size: int = 224):
    images = []
    labels = []
    cwd = Path.cwd()
    for label in dataset_dir.iterdir():
        for img_path in label.iterdir():
            # Read images
            img = cv2.imread(str(img_path))

            # Resize images
            img = cv2.resize(img, (img_size, img_size))


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
    #data_path = Path("../data") / "Plants_2"
    #train_dir = data_path / "train"
    #valid_dir = data_path / "valid"
    #test_dir = data_path / "test"
#
    #for label in test_dir.glob("*/"):
    #    for img_path in label.glob("*.jpg"):
    #        # Read images
    #        img = cv2.imread(str(img_path))
#
    #cv2.imshow(img_path.name, img)

    img = cv2.imread("../data/Plants_2/test/Alstonia Scholaris diseased (P2a)/0014_0006.JPG")
    cv2.imshow("0014_0006.JPG", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #train_loader, valid_loader, test_loader = get_dataloaders(data_path)

    #print(f"Loaded {len(train_loader.dataset)} training images.")
    #print(f"Loaded {len(valid_loader.dataset)} validation images.")
    #print(f"Loaded {len(test_loader.dataset)} test images.")
