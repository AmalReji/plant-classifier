import json
import time
from pathlib import Path

import pandas as pd
import xgboost as xgb
import numpy as np
import sys

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
#from db_utils import StarSchemaDB


def train_xgboost(train_X, train_y, objective='multi:softprob', num_class=None, eval_metric='mlogloss', n_jobs=1,
                  verbosity=0, n_estimators=100, max_depth=3):
    """ Train an XGBoost model on the provided training data.

    Args:
        train_X (np.ndarray): Training features.
        train_y (np.ndarray): Training labels.
        objective (str): Objective to use for training.
        num_class (int): Number of classes.
        eval_metric (str): Eval metric to use.
        n_jobs (int): Number of jobs to run in parallel.
        verbosity (int): Verbosity level.
            0: silent, 1: info.
        n_estimators (int): Number of estimators.
        max_depth (int): Maximum depth of the model.
    """

    if num_class is None:
        num_class = len(set(train_y))

    if verbosity == 1:
        print(f"Training XGBoost model with {num_class} classes")
        print(f"Training on {len(train_y)} samples with {train_X.shape[1]} features.")

    # Train xgboost model using the extracted features
    start_time = time.time()
    xgb_model = xgb.XGBClassifier(
        objective=objective,
        num_class=num_class,  # e.g., 22 classes
        eval_metric=eval_metric,
        n_jobs=n_jobs,
        verbosity=verbosity,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42  # For reproducibility
    )

    # Train the model
    xgb_model.fit(train_X, train_y)
    end_time = time.time()
    time_taken = end_time - start_time

    if verbosity == 1:
        print(f"Trained XGBoost model in {round(time_taken, 2)} secs.")

    return xgb_model, time_taken

if __name__ == '__main__':
    # Load pre-saved train, validation, and test sets
    train_X = np.load('train_X.npy')
    train_y = np.load('train_y.npy')
    valid_X = np.load('valid_X.npy')
    valid_y = np.load('valid_y.npy')
    test_X = np.load('test_X.npy')
    test_y = np.load('test_y.npy')

    # Read hyperparameters from JSON
    hyperparameters_file = Path(f"model_hyperparameters.json")
    if hyperparameters_file.exists():
        with open(hyperparameters_file, 'r') as f:
            hyperparameters = json.load(f)
    sampling_method = hyperparameters['sampling_method']
    objective = hyperparameters["objective"]
    num_workers = hyperparameters["num_workers"]
    n_estimators = hyperparameters["n_estimators"]
    model_name = hyperparameters["model_name"]
    max_depth = hyperparameters["max_depth"]
    eval_metric = hyperparameters["eval_metric"]
    batch_size = hyperparameters["batch_size"]

    xgb_model, training_time = train_xgboost(train_X, train_y, objective=objective, eval_metric=eval_metric,
                                             n_estimators=n_estimators, max_depth=max_depth)

    # Evaluate the model on test set
    test_pred = xgb_model.predict(test_X)
    test_accuracy = accuracy_score(test_y, test_pred)
    print("Test Set Classification Report:")
    print(classification_report(test_y, test_pred))

    # Evaluate the model on validation set
    valid_pred = xgb_model.predict(valid_X)
    valid_accuracy = accuracy_score(valid_y, valid_pred)
    print("Validation Set Classification Report:")
    print(classification_report(valid_y, valid_pred))

    # Store model results
    train_samples = len(train_y)
    test_samples = len(test_y)
    valid_samples = len(valid_y)

    model_result = (
        [sampling_method, objective, num_workers, n_estimators, model_name, max_depth, eval_metric, batch_size,
         test_accuracy, valid_accuracy, training_time, train_samples, test_samples, valid_samples]
    )

    # Unique identifier for this parameter set
    session_timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    param_id = f"{session_timestamp}_single_model"

    model_result_dict = {param_id: model_result}

    if model_result_dict:
        new_results_df = pd.DataFrame.from_dict(
            model_result_dict,
            orient='index',
            columns=(['sampling_method', 'objective', 'num_workers', 'n_estimators', 'model_name', 'max_depth',
                      'eval_metric', 'batch_size', 'test_accuracy', 'valid_accuracy', 'training_time',
                      'train_samples', 'test_samples', 'valid_samples'])
        )

    # Upload results to database
    db = ModelResultsDB()
    database_success = False
    if db.is_connected():
        database_success = db.save_model_results(new_results_df)
        if database_success:
            print("Model training results successfully saved to the database.")
            stats = db.get_summary_stats()
            if stats:
                print("Database Summary Stats:")
                for key, value in stats.items():
                    print(f"{key}: {value}")
        else:
            print("Failed to save model results to the database, falling back to CSV.")
    else:
        print("Database connection failed, falling back to CSV.")


    # Fallback to CSV if DB connection fails
    if not database_success:
        # Check if results file exists and load it
        results_file_path = Path('model_training_results.csv')
        results_file_path.parent.mkdir(exist_ok=True)

        if results_file_path.exists():
            print(f"Loading existing results from {results_file_path}")
            existing_results_df = pd.read_csv(results_file_path, index_col='param_id')
        else:
            existing_results_df = None

        # Append to existing results if file exists
        if existing_results_df is not None:
            print(f"Appending {len(new_results_df)} new results to existing {len(existing_results_df)} results")
            results_df = pd.concat([existing_results_df, new_results_df])
        else:
            results_df = new_results_df

        results_df.to_csv(results_file_path, index_label='param_id')

        print(f"Model training results saved to {results_file_path}")
        print(f"Total results in file: {len(results_df)}")