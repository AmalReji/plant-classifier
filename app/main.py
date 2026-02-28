from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from predict import load_model, single_image_prediction
import json
from config import MODEL_DIR
import logging, time, uuid
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(message)s')  # Raw JSON to standard output
logger = logging.getLogger(__name__)

app = FastAPI(title='Plant Classifier API', version='1.0.0')

app.add_middleware(CORSMiddleware, allow_origins=['*'],
                   allow_methods=['*'], allow_headers=['*'])

# Load artifacts once at startup
with open(f"{MODEL_DIR}/model_hp.json", 'r') as f:
    hyperparameters = json.load(f)
with open(f"{MODEL_DIR}/metadata.json", 'r') as f:
    metadata = json.load(f)
xgb_model = load_model()

cnn_name = hyperparameters['model_name']
CLASS_NAMES = metadata['class_names']


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())[:8]  # Short unique ID for logging
    start_time = time.time()

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='Invalid file type. Please upload an image.')

    if xgb_model is None:
        raise HTTPException(status_code=500, detail='Model is not loaded.')

    contents = await file.read()
    label, pred_idx, confidence = single_image_prediction(image_bytes=contents, cnn_name=cnn_name,
                                                                  xgb_model=xgb_model, class_names=CLASS_NAMES)

    latency_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(json.dumps({
        'request_id': request_id,
        'timestamp': datetime.now(tz=timezone.utc).isoformat(),
        'cnn': cnn_name,
        'prediction': label,
        'confidence': confidence,
        'latency_ms': latency_ms,
        'filename': file.filename
    }))

    return {
        'prediction': label,
        'class_index': pred_idx,
        'confidence': confidence,
        'request_id': request_id,
        'latency_ms': latency_ms}
