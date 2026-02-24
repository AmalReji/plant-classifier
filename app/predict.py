import io
import numpy as np
#import torch
from PIL import Image
#from torchvision.models import get_weight, get_model
import xgboost as xgb
from config import MODEL_DIR

def load_model():
    try:
        model = xgb.XGBClassifier()
        model.load_model(f"{MODEL_DIR}/model.json")
        # model = joblib.load(f"{MODEL_DIR}/model.joblib")
        print(f"Model loaded successfully from {MODEL_DIR}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def single_feature_extraction(image: Image.Image, cnn_name: str) -> np.ndarray:
    weights = get_weight(f"{cnn_name}_Weights.DEFAULT")
    preprocess = weights.transforms()

    cnn = get_model(cnn_name, weights="DEFAULT")
    cnn = torch.nn.Sequential(*list(cnn.children())[:-1])
    cnn.eval()

    # Model specific preprocessing
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        features = cnn(tensor)
        features = features.view(features.size(0), -1)  # Flatten (1, 2048, 1, 1) -> (1, 2048)

    return features.numpy()

def single_image_prediction(image_bytes, cnn_name: str, xgb_model, class_names: list) -> int:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    features = single_feature_extraction(image, cnn_name)

    prediction = xgb_model.predict(features)
    pred_idx = int(xgb_model.predict(features)[0])
    confidence = float(xgb_model.predict_proba(features)[0][pred_idx])

    label = class_names[pred_idx] if class_names else str(pred_idx)

    return {
        "prediction": label,
        "class_index": pred_idx,
        "confidence": round(confidence, 4)
    }


