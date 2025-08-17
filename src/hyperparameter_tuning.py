import json
import subprocess
import tempfile
from itertools import product
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

    # Create combinations for feature extraction (outer loop)
    feature_combinations = list(product(
        feature_extraction_params['model_name'],
        preprocess_params['sampling_method'],
        preprocess_params['batch_size'],
        preprocess_params['num_workers']
    ))

    # Create combinations for model training (inner loop)
    model_combinations = list(product(
        model_params['objective'],
        model_params['eval_metric'],
        model_params['n_estimators'],
        model_params['max_depth']
    ))

    best_accuracy = 0.0
    best_params = None
    training_order = 0
    trained_models = {}

    # Define the order of columns for results DataFrame
    column_order = list(param_grid[0].keys())

    # Generate session timestamp for unique identification
    session_timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    total_combinations = len(feature_combinations) * len(model_combinations)
    print(f"Total parameter combinations to test: {total_combinations}")
    print(f"Feature extraction combinations: {len(feature_combinations)}")
    print(f"Model training combinations per feature set: {len(model_combinations)}")

    # Outer loop: Extract features once for each model/sampling combination
    for feat_idx, (model_name, sampling_method, batch_size, num_workers) in enumerate(feature_combinations, 1):

        print(f"\n{'=' * 60}")
        print(f"Feature Extraction {feat_idx}/{len(feature_combinations)}")
        print(f"Model: {model_name}, Sampling: {sampling_method}")
        print(f"Batch size: {batch_size}, Num workers: {num_workers}")
        print(f"{'=' * 60}")

        # Step 1: Feature extraction (once per model/sampling combination)
        try:
            print("Extracting features...")
            feature_extraction_cmd = [
                'python', 'feature_extraction_subprocess.py',
                '--model_name', model_name,
                '--batch_size', str(batch_size),
                '--num_workers', str(num_workers),
                '--sampling_method', sampling_method
            ]

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

        except Exception as e:
            print(f"Error in feature extraction: {e}")
            continue


        # Inner loop: Train models with different XGBoost parameters using the same features
        for model_idx, (objective, eval_metric, n_estimators, max_depth) in enumerate(model_combinations, 1):
            training_order += 1
            print(f"\n--- Training iteration {training_order}/{total_combinations} ---")
            print(f"    Model training {model_idx}/{len(model_combinations)} for current feature set")
            print(f"    XGBoost params: n_estimators={n_estimators}, max_depth={max_depth}")

            # Create full parameter dictionary
            params = {
                'sampling_method': sampling_method,
                'batch_size': batch_size,
                'num_workers': num_workers,
                'model_name': model_name,
                'objective': objective,
                'eval_metric': eval_metric,
                'n_estimators': n_estimators,
                'max_depth': max_depth
            }

            # Unique identifier for this parameter set
            param_id = f"{session_timestamp}_iter_{training_order}"

            try:
                # Step 2: Model training using the pre-extracted features
                print("    Training model...")
                model_training_cmd = [
                    'python', 'model_training_subprocess.py',
                    '--objective', objective,
                    '--eval_metric', eval_metric,
                    '--n_estimators', str(n_estimators),
                    '--max_depth', str(max_depth),
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

                    print(f"    Model training completed with return code: {result.returncode}")

                    if result.returncode != 0:
                        print(f"    Model training failed with return code: {result.returncode}")
                        print(f"    STDERR: {stderr_content[:1000]}...")  # Show first 1000 chars of error
                        continue
                    else:
                        # Show last few lines of stdout for confirmation
                        stdout_lines = stdout_content.strip().split('\n')
                        if stdout_lines:
                            print(f"    Last output: {stdout_lines[-1]}")

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

                    print(f"    Test Set Accuracy: {test_accuracy:.4f}")
                    print(f"    Validation Set Accuracy: {valid_accuracy:.4f}")
                    print(f"    Training Time: {training_time:.2f} seconds")

                    if test_accuracy > best_accuracy:
                        best_accuracy = test_accuracy
                        best_params = params.copy()

                    print(f"    Current Best Test Accuracy: {best_accuracy:.4f} with parameters: {best_params}")

                    # Store model results
                    trained_model = (
                            [params[col] for col in column_order] +
                            [test_accuracy, valid_accuracy, training_time, train_samples, test_samples, valid_samples]
                    )
                    trained_models[param_id] = trained_model

                    # Clean up temporary files
                    results_file.unlink()

                else:
                    print(f"    Results file {results_file} not found. Skipping this iteration.")
                    continue

            except Exception as e:
                print(f"    Error in model training iteration {training_order}: {e}")
                continue

    # Save overall results
    print(f"\n{'=' * 60}")
    print(f"HYPERPARAMETER TUNING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Best parameters: {best_params}")
    print(f"Best test accuracy: {best_accuracy:.4f}")

    # Check if results file exists and load it
    results_file_path = Path('hyperparameter_tuning_results.csv')
    results_file_path.parent.mkdir(exist_ok=True)

    if results_file_path.exists():
        print(f"Loading existing results from {results_file_path}")
        existing_results_df = pd.read_csv(results_file_path, index_col='param_id')
    else:
        existing_results_df = None

    if trained_models:
        new_results_df = pd.DataFrame.from_dict(
            trained_models,
            orient='index',
            columns=(column_order +
                     ['test_accuracy', 'valid_accuracy', 'training_time',
                      'train_samples', 'test_samples', 'valid_samples'])
        )

        # Append to existing results if file exists
        if existing_results_df is not None:
            print(f"Appending {len(new_results_df)} new results to existing {len(existing_results_df)} results")
            results_df = pd.concat([existing_results_df, new_results_df])
        else:
            results_df = new_results_df

        results_df.to_csv(results_file_path, index_label='param_id')

        print(f"Hyperparameter tuning results saved to {results_file_path}")
        print(f"Total results in file: {len(results_df)}")


        # for testing
        new_results_df.to_csv('latest_hyperparameter_tuning_results.csv', index_label='param_id')

if __name__ == '__main__':
    hyperparameter_tuning()
