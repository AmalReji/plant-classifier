import time

import xgboost as xgb
import numpy as np
import sys

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

def train_xgboost(train_X, train_y, valid_X, valid_y, test_X, test_y, objective='multi:softmax',
                num_class=None, eval_metric='mlogloss', n_jobs=1, verbosity=1, n_estimators=100,
                max_depth=3):
    """ Train an XGBoost model on the provided training data.

    Args:
        train_X (np.ndarray): Training features.
        train_y (np.ndarray): Training labels.
        valid_X (np.ndarray): Validation features.
        valid_y (np.ndarray): Validation labels.
        test_X (np.ndarray): Test features.
        test_y (np.ndarray): Test labels.
        objective (str): Objective to use for training.
        num_class (int): Number of classes.
        eval_metric (str): Eval metric to use.
        n_jobs (int): Number of jobs to run in parallel.
        verbosity (int): Verbosity level.
        n_estimators (int): Number of estimators.
        max_depth (int): Maximum depth of the model.
    """

    if num_class is None:
        num_class = len(set(train_y))

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
        max_depth=max_depth
    )

    # Train the model
    xgb_model.fit(train_X, train_y)
    end_time = time.time()
    print(f"Trained XGBoost model in {end_time - start_time} secs.")

    return xgb_model

if __name__ == '__main__':
    # Load pre-saved train, validation, and test sets
    train_X = np.load('train_X.npy')
    train_y = np.load('train_y.npy')
    valid_X = np.load('valid_X.npy')
    valid_y = np.load('valid_y.npy')
    test_X = np.load('test_X.npy')
    test_y = np.load('test_y.npy')

    xgb_model = train_xgboost(train_X, train_y, valid_X, valid_y, test_X, test_y)

    # Evaluate the model on test set
    y_pred_test = xgb_model.predict(test_X)
    print("Test Set Classification Report:")
    print(classification_report(test_y, y_pred_test))

    # Evaluate the model on validation set
    y_pred_valid = xgb_model.predict(valid_X)
    print("Validation Set Classification Report:")
    print(classification_report(valid_y, y_pred_valid))
