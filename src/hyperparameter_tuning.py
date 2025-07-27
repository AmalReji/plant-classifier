from data_preprocessing import preprocess_images
from extract_features import extract_features
from train_model import train_xgboost

from sklearn.model_selection import ParameterGrid
from pathlib import Path
import pandas as pd

# Choosing the best parameters for data preprocessing, feature extraction, and model training
def hyperparameter_tuning():
    """ Perform hyperparameter tuning for data preprocessing, feature extraction, and model training. """

    # Define parameter grid for data preprocessing
    preprocess_params = {
        'sampling_method': ['over', 'under', 'none'],  # Options: "over", "under", "none"
        'batch_size': [32],
        'num_workers': [4]
    }

    # Define parameter grid for feature extraction
    feature_extraction_params = {
        'model_name': ['ResNet50', 'EfficientNet_B0']
    }

    # Define parameter grid for model training
    model_params = {
        'objective': ['multi:softmax'],
        'eval_metric': ['mlogloss'],
        'n_estimators': [100, 200],
        'max_depth': [3, 5]
    }

    # Create a grid of all parameters
    param_grid = ParameterGrid({
        **preprocess_params,
        **feature_extraction_params,
        **model_params
    })

    best_accuracy = 0.0
    best_params = None

    training_order = 0
    trained_models = {}
    for params in param_grid:
        training_order += 1
        print(f"\nTraining iteration {training_order}/{len(param_grid)}")
        print(f"Testing parameters: {params}")

        # Preprocess images
        train_loader = preprocess_images(
            dataset_dir=Path('../data/Plants_2/train'),
            model_name=params['model_name'],
            batch_size=params['batch_size'],
            num_workers=params['num_workers'],
            sampling_method=params['sampling_method']

        )

        valid_loader = preprocess_images(
            dataset_dir=Path('../data/Plants_2/valid'),
            model_name=params['model_name'],
            batch_size=params['batch_size'],
            num_workers=params['num_workers'],
            sampling_method= 'none'  # No balancing for validation set
        )

        test_loader = preprocess_images(
            dataset_dir=Path('../data/Plants_2/test'),
            model_name=params['model_name'],
            batch_size=params['batch_size'],
            num_workers=params['num_workers'],
            sampling_method= 'none' # No balancing for test set
        )

        # Extract features
        train_X, train_y = extract_features(train_loader, params['model_name'])
        valid_X, valid_y = extract_features(valid_loader, params['model_name'])
        test_X, test_y = extract_features(test_loader, params['model_name'])

        # Train the model
        xgb_model = train_xgboost(
            train_X=train_X,
            train_y=train_y,
            objective=params['objective'],
            eval_metric=params['eval_metric'],
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth']
        )

        # Evaluate the model
        y_pred_test = xgb_model.predict(test_X)
        accuracy = (y_pred_test == test_y).mean()
        print(f"Test Set Accuracy: {accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

        print(f"Current Best Accuracy: {best_accuracy:.4f} with parameters: {best_params}")

        trained_model = list(params.values()) + [accuracy]
        trained_models[training_order] = trained_model

    print(f"Best parameters: {best_params}")
    print(f"Best accuracy: {best_accuracy:.4f}")

    pd.DataFrame.from_dict(trained_models, orient='index', columns=list(params.keys()) + ['accuracy']).to_csv('hyperparameter_tuning_results.csv')

    # Save the best model parameters
    with open('best_params.txt', 'w') as f:
        f.write(str(best_params))


if __name__ == '__main__':
    hyperparameter_tuning()



