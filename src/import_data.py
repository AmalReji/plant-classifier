from collections import Counter

import kagglehub
from pathlib import Path
import shutil

# Download plant leaves dataset
path_plant = Path(kagglehub.dataset_download("csafrit2/plant-leaves-for-image-classification"))

print("Plant leaves path:", path_plant)

# Move files to destination
destination = Path("../data")
destination.mkdir(parents=True, exist_ok=True)  # Mimics bash `mkdir -p`
shutil.copytree(path_plant, destination, dirs_exist_ok=True)  # Existing files in destination will be overwritten

print(f"Dataset moved to {destination}")

# Download jackfruit leaves dataset
path_jack = Path(kagglehub.dataset_download("shuvokumarbasak4004/jackfruit-leaf-diseases"))

print("Jackfruit leaves path:", path_jack)

# Wrangle Jackfruit data into same format as previous dataset
splits = ["train", "test", "validation", "valid"]

def normalize_split(name: str) -> str:
    name = name.lower()
    if "train" in name:
        return "train"
    if "valid" in name:
        return "valid"
    if "test" in name:
        return "test"
    return "train" # fallback

for img_path in path_jack.rglob("*.jpg"):
    # infer split from parent folders
    parts = [p.lower() for p in img_path.relative_to(path_jack).parts]
    for p in parts:
        if p in splits:
            split = normalize_split(p)
            break

    class_raw = img_path.parent.name
    match class_raw:
        case "Algal_Leaf_Spot_of_Jackfruit" | "Black_Spot_of_Jackfruit":
            class_name = "Jackfruit diseased"
        case "Healthy_Leaf_of_Jackfruit":
            class_name = "Jackfruit healthy"
        case _:
            class_name = class_raw
    dest_dir = destination / "Plants_2" / split / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(img_path, dest_dir / img_path.name)

# Verification
for split in ["train","valid","test"]:
    counts = Counter(p.parent.name for p in (destination/"Plants_2"/split).rglob("*.*"))
    print(split, len(counts), "classes,", sum(counts.values()), "images")