# Plant Classifier

Classify healthy vs unhealthy leaves using ML.

## Data

Due to size constraints, the dataset is not uploaded to GitHub.

To download the dataset, run:

```bash
cd src
python import_data.py
```

# Plant Classifier

A machine learning pipeline for classifying healthy vs unhealthy plant leaves using transfer learning with CNN feature extraction and XGBoost classification.

## Architecture

The project uses a hybrid approach:
- **Feature Extraction**: Pre-trained ResNet50 model (without final classification layer)
- **Classification**: XGBoost classifier trained on extracted features
- **Data Pipeline**: PyTorch DataLoaders with class balancing via WeightedRandomSampler
- **Database**: PostgreSQL star schema for training experiment tracking (coming soon)

## Project Structure

```
plant-classifier/
├── src/
│   ├── import_data.py          # Download dataset from Kaggle
│   ├── data_preprocessing.py   # Image preprocessing and DataLoader creation
│   └── train_model.py          # Feature extraction and model training
├── app/                        # Application deployment (TBD)
├── tests/                      # Unit tests
├── data/                       # Dataset (auto-downloaded, gitignored)
├── requirements.txt            # Python dependencies
└── main.py                     # Main application entry point (TBD)
```

## Requirements

- Python 3.12.10
- See `requirements.txt` for complete dependency list
- Key libraries: PyTorch, torchvision, XGBoost, scikit-learn, OpenCV

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

## Usage

### Training the Model (~80 mins on CPU)

```bash
cd src
python train_model.py
```

This will:
1. Load and preprocess images from `data/Plants_2/train`, `valid`, and `test` directories
2. Extract features using pre-trained ResNet50
3. Train an XGBoost classifier on the extracted features
4. Evaluate performance on the test set

### Key Features

- **Class Balancing**: Uses WeightedRandomSampler to handle imbalanced datasets
- **Transfer Learning**: Leverages ResNet50 pre-trained on ImageNet for feature extraction
- **Efficient Processing**: Multi-worker DataLoaders for faster image loading
- **Model Flexibility**: Easy to swap different pre-trained models via `model_name` parameter

## Dataset

The project uses the [Plant Leaves for Image Classification](https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification) dataset from Kaggle, which contains images organized by plant health status.

**Note**: Due to size constraints, the dataset is not included in the repository and must be downloaded using the provided script.

## Experiment Tracking (Coming Soon)

A PostgreSQL star schema database will be integrated to track:
- Training hyperparameters
- Model performance metrics
- Feature extraction configurations
- Dataset versions and splits
- Experiment timestamps and metadata

This will enable:
- Historical comparison of model iterations
- Hyperparameter optimization tracking
- Reproducible experiment management
- Performance trend analysis

## Model Performance

The current pipeline extracts high-dimensional features from ResNet50's penultimate layer and trains an XGBoost classifier. Performance metrics are displayed via scikit-learn's classification report.

## Future Enhancements

- [x] Apply correct preprocessing depending on the chosen CNN
- [x] Automated class balancing
- [ ] PostgreSQL experiment tracking database
- [ ] Web application for model deployment (`app/` directory)
- [ ] Hyperparameter optimization pipeline
- [ ] Model serving API
- [ ] Docker containerization
- [ ] CI/CD pipeline with automated testing

## Contributing

1. Ensure Python 3.12.10 is installed
2. Follow the setup instructions above
3. Run tests before submitting PRs (test framework TBD)
4. Update this README for any architectural changes