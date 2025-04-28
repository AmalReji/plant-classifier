import kagglehub
from pathlib import Path
import shutil

# Download latest version
path = kagglehub.dataset_download("csafrit2/plant-leaves-for-image-classification")

print("Path to dataset files:", path)

# Move files to destination
destination = Path("../data")
destination.mkdir(parents=True, exist_ok=True)  # Mimics bash `mkdir -p`
shutil.copytree(path, destination, dirs_exist_ok=True)  # Existing files in destination will be overwritten

print(f"Dataset moved to {destination}")
