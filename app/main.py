from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from predict import single_image_prediction
from predict import load_model
import json
from config import MODEL_DIR, MODEL_VERSION

app = FastAPI(title="Plant Classifier API", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Load artifacts once at startup
with open(f"{MODEL_DIR}/model_hp.json", 'r') as f:
    hyperparameters = json.load(f)
with open(f"{MODEL_DIR}/metadata.json", 'r') as f:
    metadata = json.load(f)
xgb_model = load_model()

cnn_name = hyperparameters['model_name']

CLASS_NAMES = metadata['class_names']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    contents = await file.read()
    prediction_dict = single_image_prediction(image_bytes=contents,
                                                                  cnn_name=cnn_name,
                                                                  xgb_model=xgb_model,
                                                                  class_names=CLASS_NAMES)
    return prediction_dict