import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import xgboost as xgb
from config import MODEL_DIR


def load_model():
    try:
        model = xgb.XGBClassifier()
        model.load_model(f"{MODEL_DIR}/model.json")
        print(f"Model loaded successfully from {MODEL_DIR}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def single_feature_extraction(image: Image.Image, cnn_name: str) -> np.ndarray:
    script_path = Path(__file__).resolve().parent / 'torch_feature_subprocess.py'
    python_executable = sys.executable

    # Save the in-memory image to a temporary file so the subprocess can read it.
    # NamedTemporaryFile gives us a unique path (`temp_image_file.name`) that exists
    # only for the lifetime of this context manager.
    with tempfile.NamedTemporaryFile(suffix='.png') as temp_image_file:
        image.save(temp_image_file.name, format='PNG')
        cmd = [
            # Use the same Python interpreter running the API process so the
            # subprocess sees the same virtualenv/site-packages.
            python_executable,
            str(script_path),
            '--image-path',
            temp_image_file.name,
            '--cnn-name',
            cnn_name
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise RuntimeError(
            f"Feature extraction subprocess failed with code {result.returncode}: {result.stderr.strip()}"
        )

    payload = json.loads(result.stdout)
    return np.array(payload['features'], dtype=np.float32)


def single_image_prediction(image_bytes, cnn_name: str, xgb_model, class_names: list) -> int:
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    features = single_feature_extraction(image, cnn_name)

    pred_idx = int(xgb_model.predict(features)[0])
    confidence = float(xgb_model.predict_proba(features)[0][pred_idx])

    label = class_names[pred_idx] if class_names else str(pred_idx)

    return label, pred_idx, round(confidence, 4)
