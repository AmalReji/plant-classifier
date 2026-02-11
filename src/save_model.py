import json
from pathlib import Path

import joblib
import numpy as np

from db_utils import StarSchemaDB
from data_preprocessing import preprocess_images
from extract_features import extract_features
from train_model import train_xgboost

db = StarSchemaDB()

if db.is_connected():
    best_model_params_df = db.get_best_models(num_models=1)
    best_model_params = best_model_params_df.iloc[0].to_dict()

    if best_model_params is not None:
        # Set up paths
        root_dir = Path.cwd().parent
        data_path = Path.joinpath(root_dir, "data", "Plants_2")
        train_dir = Path.joinpath(data_path, "train")
        valid_dir = Path.joinpath(data_path, "valid")
        test_dir = Path.joinpath(data_path, "test")

        try:
            # Preprocess images and create DataLoaders
            train_loader = preprocess_images(
                dataset_dir=train_dir,
                model_name=best_model_params["model_name"],
                batch_size=best_model_params["batch_size"],
                num_workers=best_model_params["num_workers"],
                sampling_method=best_model_params["sampling_method"]
            )

            # Extract features from the datasets
            train_X, train_y = extract_features(dataloader=train_loader, model_name=best_model_params["model_name"])

            # Save the extracted features and labels
            np.save('train_X.npy', train_X)
            np.save('train_y.npy', train_y)

            print("Feature extraction completed and saved to .npy files.")

        except Exception as e:
            print(f"Error during feature extraction: {e}")
            raise

    else:
        print("Failed to find best model.")
else:
    print("Database connection failed.")

xgb_model, train_time = train_xgboost(train_X, train_y,
                                      objective=best_model_params["objective"],
                                      eval_metric=best_model_params["eval_metric"],
                                      n_estimators=best_model_params["n_estimators"],
                                      max_depth=best_model_params["max_depth"]
                                      )
Path.joinpath(root_dir, 'artefacts').mkdir(parents=True, exist_ok=True)
joblib.dump(xgb_model, '../artefacts/xgb_model.joblib')
json.dump(best_model_params, open('../artefacts/xgb_model.json', 'w'), indent=2)
print("Model saved to artefacts/")
