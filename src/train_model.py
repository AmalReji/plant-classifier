import time

import xgboost as xgb
import numpy as np
import sys

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

if __name__ == '__main__':
    # Load pre-saved train_X.npy/train_y.npy, cast label→int32
    train_X = np.load('train_X.npy')
    train_y = np.load('train_y.npy')
    valid_X = np.load('valid_X.npy')
    valid_y = np.load('valid_y.npy')
    test_X = np.load('test_X.npy')
    test_y = np.load('test_y.npy')

    print(f"Training on {len(train_y)} samples with {train_X.shape[1]} features.")
    # print(f"train_X.dtype={train_X.dtype}, train_X.shape={train_X.shape}")
    # print(f"train_y.dtype={train_y.dtype}, train_y.shape={train_y.shape}")

    # # Reduce dimensionality using PCA
    # print(f"Original number of features: {train_X.shape[1]}")
    # pca = PCA(n_components=10)  # Try fewer components
    # train_X_reduced = pca.fit_transform(train_X)
    # print("Reduced features to", train_X_reduced.shape[1], "components.")

    # Train xgboost model using the extracted features
    start_time = time.time()
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(set(train_y)),  # e.g., 22 classes
        eval_metric='mlogloss',
        n_jobs=1,
        verbosity=1,
        n_estimators=50,
        max_depth=4
    )

    # Train the model
    xgb_model.fit(train_X, train_y)
    end_time = time.time()
    print(f"Trained XGBoost model in {end_time - start_time} secs.")

    # Evaluate the model on validation set
    # y_pred = xgb_model.predict(valid_X)
    # print(classification_report(valid_y, y_pred))

    # Evaluate the model on test set
    y_pred_test = xgb_model.predict(test_X)
    print(classification_report(test_y, y_pred_test))

    # Evaluate the model on validation set
    y_pred_valid = xgb_model.predict(valid_X)
    print(classification_report(test_y, y_pred_valid))
