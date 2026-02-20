import argparse
from datetime import datetime, timezone
import json
import tempfile
import time
from pathlib import Path
import subprocess
import joblib
import numpy as np
import pandas as pd

from db_utils import StarSchemaDB
from train_model import train_xgboost

# Handle the --version argument
parser = argparse.ArgumentParser(description='Promote a specific model version to production.')
parser.add_argument("--version", type=str, default=None, required=True, help="Version to promote.")
args = parser.parse_args()
version = args.version

db = StarSchemaDB()



if db.is_connected():
    # Step 1: Find best model parameters from db
    best_model_params_df = db.get_best_models(num_models=1)
    best_model_params = best_model_params_df.iloc[0].to_dict()
    if best_model_params is not None:

        # Step 2: Feature extraction
        try:
            print("Extracting features...")
            feature_extraction_cmd = [
                'python', "feature_extraction_subprocess.py",
                '--model_name', best_model_params["model_name"],
                '--batch_size', str(best_model_params["batch_size"]),
                '--num_workers', str(best_model_params["num_workers"]),
                '--sampling_method', best_model_params["sampling_method"]
            ]
            with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as stdout_file, \
                    tempfile.TemporaryFile(mode='w+', encoding='utf-8') as stderr_file:

                extract_start = time.time()
                result = subprocess.run(feature_extraction_cmd,
                                        stdout=stdout_file,
                                        stderr=stderr_file,
                                        text=True,
                                        cwd=Path.cwd())
                feature_extract_time = time.time() - extract_start

                # Read the output for debugging if needed
                stdout_file.seek(0)
                stderr_file.seek(0)
                stdout_content = stdout_file.read()
                stderr_content = stderr_file.read()

            print(f"Feature extraction completed with return code: {result.returncode}")
            print(f"Feature extraction Time: {feature_extract_time:.2f} seconds")

            if result.returncode != 0:
                print(f"Feature extraction failed with return code: {result.returncode}")
                print(f"STDERR: {stderr_content[:1000]}...")  # Show first 1000 chars of error
            else:
                # Show last few lines of stdout for progress confirmation
                stdout_lines = stdout_content.strip().split('\n')
                if stdout_lines:
                    print(f"Last output: {stdout_lines[-1]}")

        except Exception as e:
            print(f"Error in feature extraction: {e}")

        # Step 3: Model training using the extracted features
        try:
            print("Training model...")
            # Load pre-saved train, validation, and test sets
            train_X = np.load('train_X.npy')
            train_y = np.load('train_y.npy')

            xgb_model, train_time = train_xgboost(train_X, train_y,
                                                  objective=best_model_params["objective"],
                                                  eval_metric=best_model_params["eval_metric"],
                                                  n_estimators=best_model_params["n_estimators"],
                                                  max_depth=best_model_params["max_depth"]
                                                  )
        except Exception as e:
            print(f"Error in model training: {e}")
    else:
        print("Failed to find best model.")
else:
    print("Database connection failed.")

# Step 4: Promote model along with hyperparameters and metadata
train_dir = Path('../data/Plants_2/train')
class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
metadata = {"version": version,
            "promoted_at": datetime.now(tz=timezone.utc).isoformat(),
            "class_names": class_names}

cleaned_model_params = {
    k: (v.isoformat() if isinstance(v, pd.Timestamp) else v)
    for k, v in best_model_params.items()
}

root_dir = Path.cwd().parent
Path.joinpath(root_dir, f'app/models/model_v{version}').mkdir(parents=True, exist_ok=True)
joblib.dump(xgb_model, f'../app/models/model_v{version}/model.joblib')
json.dump(cleaned_model_params, open(f'../app/models/model_v{version}/model_hp.json', 'w'), indent=2)
json.dump(metadata, open(f'../app/models/model_v{version}/metadata.json', 'w'), indent=2)
print(f"Model saved to app/models/model_v{version}/")
