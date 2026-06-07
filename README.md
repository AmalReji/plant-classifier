---
title: Plant Classifier
emoji: 🌱
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
license: mit
short_description: Classify plant leaves on plant name and healthy vs diseased.
---
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Plant Classifier

A machine learning pipeline for classifying healthy vs unhealthy plant leaves using transfer learning with CNN feature extraction and XGBoost classification.

## Architecture

The project uses a hybrid approach:
- **Feature Extraction**: Pre-trained ResNet50 model (without final classification layer)
- **Data Pipeline**: PyTorch DataLoaders with class balancing via WeightedRandomSampler
- **Classification**: XGBoost classifier trained on extracted features
- **Database**: PostgreSQL star schema for training experiment tracking
- **Deployment**: HF Spaces web application for model inference

## Project Structure

```
plant-classifier/
├── .github/
│   └── workflows               
│       └── hf-deploy.yml                 # CI/CD pipeline for testing and deployment to Hugging Face Spaces
├── tests/                                 
│   └── test_model.py                     # Unit tests
├── data/                                 # Dataset (downloaded with import_data.py, gitignored)
├── src/          
│   ├── import_data.py                    # Download dataset from Kaggle and arrange in data/ directory
│   ├── hyperparameter_tuning.py          # Hyperparameter tuning for feature extraction and model training (saves results to DB)
│   ├── feature_extraction_subprocess.py  # Image preprocessing and DataLoader creation
│   ├── data_preprocessing.py             # [redundant?] Image preprocessing and DataLoader creation
│   ├── extract_features.py               # [redundant?] Feature extraction using pre-trained CNN
│   └── train_model.py                    # [redundant?] Train XGBoost classifier on extracted features
├── app/                                  # Application deployment
│   ├── models                            
│     ├── model_v1                        
│     ├── model_v2                        
│     └── ...                       
│   ├── config.py                         # reads model_version.env
│   ├── main.py                           # Gradio app for model inference
│   ├── predict.py                        # Inference script for loading model and making predictions
│   ├── torch_feature_subprocess.py       # Subprocess to handle feature extraction for predict.py
│   └── requirements_inference.txt        # Dependencies for inference (lighter than full requirements.txt)
├── requirements.txt                      # Python dependencies
├── main.py                               # [redundant]
└── .env                                  # (create this yourself) DB credentials: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
```

## Requirements

- Python 3.12.10
- See `requirements.txt` for complete dependency list
- Key libraries: PyTorch, torchvision, XGBoost, scikit-learn, OpenCV

## Dataset
The dataset is a combination of two Kaggle datasets:
- [Plant Leaves for Image Classification](https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification)
- [Jackfruit Leaf Diseases](https://www.kaggle.com/datasets/shuvokumarbasak4004/jackfruit-leaf-diseases)

Due to size constraints, the dataset is not uploaded to GitHub.

To download the combined dataset, clone the repo and run:

```bash
cd src
python import_data.py
```

## Setup

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download dataset**:
   ```bash
   cd src
   python import_data.py
   ```
   This downloads the plant leaves dataset from Kaggle and organizes it in the `data/` directory.

### Key Features

- **Class Balancing**: Uses WeightedRandomSampler to handle imbalanced datasets
- **Transfer Learning**: Leverages ResNet50 pre-trained on ImageNet for feature extraction
- **Model Flexibility**: Easy to swap different pre-trained CNNs via `model_name` parameter

### Training the Model (~48 hrs on CPU)

  Modify _params dictionaries in `hyperparameter_tuning.py` to set the hyperparameter space for feature extraction and model training. Then run:
```bash
# remain in src directory
python hyperparameter_tuning.py
```

  This will:
1. Load and preprocess images from `data/Plants_2/train`, `valid`, and `test` directories
2. Extract features using a pre-trained CNN
3. Train an XGBoost classifier on the extracted features
4. Evaluate performance on the test set
5. Upload the results and model metadata to the PostgreSQL database for experiment tracking

### Promote Best Model to Production (~2 hrs on CPU)

  Update `model_version.env`'s `MODEL_VERSION` variable if there is a significant change in the model (should be an integer e.g., 1).

  The following script then identifies the best model (based on validation performance) from the database and promotes it to production:
```bash
# remain in src directory
python promote_model.py
```

## Usage

### Model Deployment

The Dockerfile reads the chosen model from `model_version.env` and copies it into the container for deployment in a gradio application.

The Dockerfile is deployed to a Hugging Face Spaces web application upon successful push to this GitHub repo's main branch. The app allows users to upload leaf images and receive predictions on plant health status.

Link to app: https://huggingface.co/spaces/your-username/plant-classifier


## Experiment Tracking

A local PostgreSQL star schema database is used to track:
- Training hyperparameters
- Model performance metrics
- Feature extraction configurations
- Dataset versions and splits
- Experiment timestamps and metadata

This enables:
- Hyperparameter optimization tracking
- Promotion of "best" model to production based on validation performance

### Credentials
Include the following variables (replacing the placeholder values) in a `.env` file (not committed to GitHub):
```
DB_HOST='localhost'
DB_PORT=5432
DB_NAME='model_db'
DB_USER='model_user'
DB_PASSWORD='postgres'
```


## Model Performance

The current pipeline extracts high-dimensional features from ResNet50's penultimate layer and trains an XGBoost classifier. Performance metrics are displayed via scikit-learn's classification report.

### Model_v1
- ResNet50 features + XGBoost classifier

### Model_v2 updates
- Added Jackfruit leaf disease dataset
- Forced selection of class balanced model in `promote_model.py` due to severe imbalance in the new dataset 

### Current Results

| Dataset    | Model_v1 Accuracy | Model_v2 Accuracy | 
|------------|-------------------|-------------------|
| Validation | 86.36%            | 99.13%            |  
| Test       | 90%               | 98.90%            | 

## Future Enhancements

- [x] Apply correct preprocessing depending on the chosen CNN
- [x] Automated class balancing
- [x] PostgreSQL experiment tracking database
- [x] Hyperparameter optimization pipeline
- [x] Model serving API
- [x] Docker containerization
- [x] Web application for model deployment (`app/` directory)
- [x] CI/CD pipeline with automated testing

## Contributing

1. Ensure Python 3.12.10 is installed
2. Follow the setup instructions above
3. Run tests before submitting PRs (see `hf-deploy.yml`)
4. Update this README for any architectural changes

### Modifying dataset
To modify the dataset (e.g., add new plant classes, augment data, etc.), make the changes in `src/import_data.py` and then run `src/hyperparameter_tuning.py` to retrain the model with the new dataset.

### Incrementing Model Versions
When making significant changes to the model architecture, training data, or hyperparameters that impact performance, increment `model_version.env`'s `MODEL_VERSION` variable to reflect the new version (e.g., from 1 to 2). This allows `src/promote_model.py` to identify and promote the new model version to production.

### Modifying Web App class names
To modify the class names displayed in the web application, update the `CLASS_NAMES` cleaning steps in `app/main.py`. 