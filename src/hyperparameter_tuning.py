import json
import subprocess
import tempfile
from pathlib import Path
import pandas as pd
from sklearn.model_selection import ParameterGrid

'''This script performs hyperparameter tuning for data preprocessing, feature extraction, and model training.'''

def hyperparameter_tuning():
    """ Perform hyperparameter tuning for data preprocessing, feature extraction, and model training. """

    # Define parameter grid for data preprocessing
    preprocess_params = {
        'sampling_method': ['over', 'under', 'none'],  # Options: "over", "under", "none"
        'batch_size': [64],
        'num_workers': [0]
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

    # Define the order of columns for results DataFrame
    column_order = list(param_grid[0].keys())

    # Generate session timestamp for unique identification
    session_timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    for params in param_grid:
        training_order += 1
        print(f"\nTraining iteration {training_order}/{len(param_grid)}")
        print(f"Testing parameters: {params}")

        # Unique identifier for this parameter set
        param_id = f"{session_timestamp}_iter_{training_order}"

        # PERFORM PREPROCESSING IN SEPARATE SCRIPT, THEN CALL USING SUBPROCESS
        try:
            # Step 1: Feature extraction using subprocess
            print("Extracting features...")
            feature_extraction_cmd = [
                'python', 'feature_extraction_subprocess.py',
                '--model_name', params['model_name'],
                '--batch_size', str(params['batch_size']),
                '--num_workers', str(params['num_workers']),
                '--sampling_method', params['sampling_method']
            ]

            # Use temporary files to avoid PIPE buffer issues
            with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as stdout_file, \
                    tempfile.TemporaryFile(mode='w+', encoding='utf-8') as stderr_file:

                result = subprocess.run(feature_extraction_cmd,
                                        stdout=stdout_file,
                                        stderr=stderr_file,
                                        text=True,
                                        cwd=Path.cwd())

                # Read the output for debugging if needed
                stdout_file.seek(0)
                stderr_file.seek(0)
                stdout_content = stdout_file.read()
                stderr_content = stderr_file.read()

                print(f"Feature extraction completed with return code: {result.returncode}")

                if result.returncode != 0:
                    print(f"Feature extraction failed with return code: {result.returncode}")
                    print(f"STDERR: {stderr_content[:1000]}...")  # Show first 1000 chars of error
                    continue
                else:
                    # Show last few lines of stdout for progress confirmation
                    stdout_lines = stdout_content.strip().split('\n')
                    if stdout_lines:
                        print(f"Last output: {stdout_lines[-1]}")

            # Step 2: Model training using subprocess
            print("Training model...")
            model_training_cmd = [
                'python', 'model_training_subprocess.py',
                '--objective', params['objective'],
                '--eval_metric', params['eval_metric'],
                '--n_estimators', str(params['n_estimators']),
                '--max_depth', str(params['max_depth']),
                '--param_id', param_id
            ]

            with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as stdout_file, \
                    tempfile.TemporaryFile(mode='w+', encoding='utf-8') as stderr_file:

                result = subprocess.run(model_training_cmd,
                                        stdout=stdout_file,
                                        stderr=stderr_file,
                                        text=True,
                                        cwd=Path.cwd())

                # Read the output
                stdout_file.seek(0)
                stderr_file.seek(0)
                stdout_content = stdout_file.read()
                stderr_content = stderr_file.read()

                print(f"Model training completed with return code: {result.returncode}")

                if result.returncode != 0:
                    print(f"Model training failed with return code: {result.returncode}")
                    print(f"STDERR: {stderr_content[:1000]}...")  # Show first 1000 chars of error
                    continue
                else:
                    # Show last few lines of stdout for confirmation
                    stdout_lines = stdout_content.strip().split('\n')
                    if stdout_lines:
                        print(f"Last output: {stdout_lines[-1]}")

            # Step 3: Read results
            results_file = Path(f"{param_id}_results.json")
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                test_accuracy = results['test_accuracy']
                valid_accuracy = results['valid_accuracy']
                training_time = results.get('training_time', 0.0)  # In case training failed
                train_samples = results['train_samples']
                test_samples = results['test_samples']
                valid_samples = results['valid_samples']

                print(f"Test Set Accuracy: {test_accuracy:.4f}")
                print(f"Validation Set Accuracy: {valid_accuracy:.4f}")
                print(f"Training Time: {training_time:.2f} seconds")

                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_params = params.copy()

                print(f"Current Best Accuracy: {best_accuracy:.4f} with parameters: {best_params}")

                # Store model results
                trained_model = (
                    [params[col] for col in column_order] +  # use consistent key order
                    [test_accuracy, valid_accuracy, training_time, train_samples, test_samples, valid_samples]
                )
                trained_models[param_id] = trained_model

                # Clean up temporary files
                results_file.unlink()

            else:
                print(f"Results file {results_file} not found. Skipping this iteration.")
                continue

        except subprocess.CalledProcessError as e:
            print(f"Subprocess error in iteration {training_order}: {e}")
            print(f"Return code: {e.returncode}")
            continue

        except Exception as e:
            print(f"Error in iteration {training_order}: {e}")
            continue

    # Save overall results
    print(f"\nBest parameters: {best_params}")
    print(f"Best accuracy: {best_accuracy:.4f}")

    if trained_models:
        results_df = pd.DataFrame.from_dict(
            trained_models,
            orient='index',
            columns= (column_order +
                      ['test_accuracy', 'valid_accuracy', 'training_time', 'train_samples', 'test_samples', 'valid_samples'])
        )

        results_df.to_csv('test/hyperparameter_tuning_results.csv', index_label='param_id')
        print("Hyperparameter tuning results saved to hyperparameter_tuning_results.csv")


if __name__ == '__main__':
    hyperparameter_tuning()
