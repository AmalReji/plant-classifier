import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import get_model, get_weight


def main():
    parser = argparse.ArgumentParser(description='Extract CNN features for a single image.')
    parser.add_argument('--image-path', type=Path, required=True, help='Path to the image file.')
    parser.add_argument('--cnn-name', type=str, required=True, help='Torchvision model name (e.g., ResNet50).')
    args = parser.parse_args()

    image = Image.open(args.image_path).convert('RGB')

    weights = get_weight(f"{args.cnn_name}_Weights.DEFAULT")
    preprocess = weights.transforms()

    cnn = get_model(args.cnn_name, weights='DEFAULT')
    cnn = torch.nn.Sequential(*list(cnn.children())[:-1])
    cnn.eval()

    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        features = cnn(tensor)
        features = features.view(features.size(0), -1)

    feature_array = features.numpy().astype(np.float32)
    print(json.dumps({'features': feature_array.tolist()}))


if __name__ == '__main__':
    main()
