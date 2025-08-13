import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from train_model import train_xgboost

'''This script trains an XGBoost model for hyperparameter tuning using pre-extracted features.'''

def main():
    parser = argparse.ArgumentParser(description='Train XGBoost model for hyperparameter tuning.')
    parser.add_argument('--objective', type=str, default='multi:softmax', help='Objective function for XGBoost.')
    parser.add_argument('--eval_metric', type=str, default='mlogloss', help='Evaluation metric for XGBoost.')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the ensemble.')
    parser.add_argument('--max_depth', type=int, default=3, help='Maximum depth of the trees.')
    parser.add_argument('--param_id', type=str, required=True, help='Parameter set ID for tracking.')

    args = parser.parse_args()

    input_dir = Path.cwd()

    print(f"Training XGBoost model with parameters: {args}")

    try:
        # Load pre-saved train, validation, and test sets
        train_X = np.load('train_X.npy')
        train_y = np.load('train_y.npy')
        valid_X = np.load('valid_X.npy')
        valid_y = np.load('valid_y.npy')
        test_X = np.load('test_X.npy')
        test_y = np.load('test_y.npy')

        print(f"Loaded features: Train: {train_X.shape}, Valid: {valid_X.shape}, Test: {test_X.shape}")

        # Train the XGBoost model
        xgb_model, training_time = train_xgboost(
            train_X=train_X,
            train_y=train_y,
            objective=args.objective,
            eval_metric=args.eval_metric,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth
        )

        # Evaluate the model on the test set
        test_pred = xgb_model.predict(test_X)
        test_accuracy = accuracy_score(test_y, test_pred)
        print(f"Test Accuracy: {test_accuracy}")

        # Evaluate the model on the validation set
        valid_pred = xgb_model.predict(valid_X)
        valid_accuracy = accuracy_score(valid_y, valid_pred)
        print(f"Validation Accuracy: {valid_accuracy}")

        # Save the model and results to JSON file
        results = {
            'test_accuracy': test_accuracy,
            'valid_accuracy': valid_accuracy,
            'training_time': training_time,
            'train_samples': len(train_y),
            'test_samples': len(test_y),
            'valid_samples': len(valid_y)
        }

        results_file = f"{args.param_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Model training results saved to {results_file}")

    except Exception as e:
        print(f"Error during model training: {e}")
        raise

if __name__ == "__main__":
    main()