"""
Star Schema Database utility functions for model training results.
This module implements a star schema with fact and dimension tables.
"""
import os
from typing import Dict, Any, Optional
import hashlib

import pandas as pd
from IPython.core.display_functions import display
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert


class StarSchemaDB:
    """Class to handle star schema database operations for model training results"""

    def __init__(self, db_config: Dict[str, Any] = None):
        """Initialise the database connection and create star schema"""
        if db_config is None:
            # Default configuration using .env
            load_dotenv()

            self.db_config = {
                'host': os.getenv('DB_HOST'),
                'port': os.getenv('DB_PORT'),
                'database': os.getenv('DB_NAME'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD')
            }
        else:
            self.db_config = db_config

        self.engine = None
        self._connect()
        self._create_star_schema()

    def _connect(self):
        """Create a SQLAlchemy engine for database operations"""
        try:
            connection_string = (f"postgresql://{self.db_config['user']}:{self.db_config['password']}@"
                               f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}")
            self.engine = create_engine(connection_string)

            # Test the connection
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))

        except Exception as e:
            print(f"Database connection failed: {e}")
            print("\nTroubleshooting tips:")
            print("1. Make sure PostgreSQL is running: brew services start postgresql@14")
            print("2. Check if database exists: psql -d model_results_db -U model_user")
            print("3. Verify credentials in DB_CONFIG")
            self.engine = None

    def is_connected(self) -> bool:
        """Check if the database connection is established"""
        return self.engine is not None

    def _create_star_schema(self):
        """Create the star schema tables if they don't exist"""
        if not self.is_connected():
            print("No database connection. Cannot create star schema.")
            return

        # Dimension tables
        create_dim_model_sql = """
        CREATE TABLE IF NOT EXISTS dim_model (
            model_id SERIAL PRIMARY KEY,
            model_name VARCHAR(100) UNIQUE NOT NULL
        );
        """

        create_dim_hyperparameters_sql = """
        CREATE TABLE IF NOT EXISTS dim_hyperparameters (
            hyperparameter_id SERIAL PRIMARY KEY,
            hyperparameter_hash VARCHAR(64) UNIQUE NOT NULL,
            objective VARCHAR(50),
            eval_metric VARCHAR(50),
            n_estimators INTEGER,
            max_depth INTEGER
        );
        """

        create_dim_preprocessing_sql = """
        CREATE TABLE IF NOT EXISTS dim_preprocessing (
            preprocessing_id SERIAL PRIMARY KEY,
            preprocessing_hash VARCHAR(64) UNIQUE NOT NULL,
            sampling_method VARCHAR(50),
            batch_size INTEGER,
            num_workers INTEGER
        );
        """

        create_dim_dataset_sql = """
        CREATE TABLE IF NOT EXISTS dim_dataset (
            dataset_id SERIAL PRIMARY KEY,
            dataset_hash VARCHAR(64) UNIQUE NOT NULL,
            train_samples INTEGER,
            valid_samples INTEGER,
            test_samples INTEGER
        );
        """

        # Fact table
        create_fact_training_sql = """
        CREATE TABLE IF NOT EXISTS fact_training_results (
            result_id SERIAL PRIMARY KEY,
            param_id VARCHAR(100) UNIQUE NOT NULL,
            model_id INTEGER REFERENCES dim_model(model_id),
            hyperparameter_id INTEGER REFERENCES dim_hyperparameters(hyperparameter_id),
            preprocessing_id INTEGER REFERENCES dim_preprocessing(preprocessing_id),
            dataset_id INTEGER REFERENCES dim_dataset(dataset_id),
            test_accuracy FLOAT,
            valid_accuracy FLOAT,
            training_time FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        # Create indexes for better query performance
        create_indexes_sql = """
        CREATE INDEX IF NOT EXISTS idx_fact_test_accuracy ON fact_training_results(test_accuracy);
        CREATE INDEX IF NOT EXISTS idx_fact_valid_accuracy ON fact_training_results(valid_accuracy);
        CREATE INDEX IF NOT EXISTS idx_fact_training_time ON fact_training_results(training_time);
        CREATE INDEX IF NOT EXISTS idx_fact_model_id ON fact_training_results(model_id);
        """

        try:
            with self.engine.connect() as connection:
                connection.execute(text(create_dim_model_sql))
                connection.execute(text(create_dim_hyperparameters_sql))
                connection.execute(text(create_dim_preprocessing_sql))
                connection.execute(text(create_dim_dataset_sql))
                connection.execute(text(create_fact_training_sql))
                connection.execute(text(create_indexes_sql))
                connection.commit()
                print("Star schema tables created successfully.")
        except Exception as e:
            print(f"Failed to create star schema: {e}")

    def _get_or_create_model_id(self, model_name: str, connection) -> int:
        """Get existing model_id or create new entry"""
        result = connection.execute(
            text("SELECT model_id FROM dim_model WHERE model_name = :model_name"),
            {"model_name": model_name}
        )
        row = result.fetchone()
        if row:
            return row[0]

        result = connection.execute(
            text("INSERT INTO dim_model (model_name) VALUES (:model_name) RETURNING model_id"),
            {"model_name": model_name}
        )
        return result.fetchone()[0]

    def _get_or_create_hyperparameter_id(self, params: Dict, connection) -> int:
        """Get existing hyperparameter_id or create new entry"""
        # Create hash of hyperparameters
        param_str = f"{params['objective']}_{params['eval_metric']}_{params['n_estimators']}_{params['max_depth']}"
        param_hash = hashlib.sha256(param_str.encode()).hexdigest()

        result = connection.execute(
            text("SELECT hyperparameter_id FROM dim_hyperparameters WHERE hyperparameter_hash = :hash"),
            {"hash": param_hash}
        )
        row = result.fetchone()
        if row:
            return row[0]

        result = connection.execute(
            text("""
                INSERT INTO dim_hyperparameters 
                (hyperparameter_hash, objective, eval_metric, n_estimators, max_depth) 
                VALUES (:hash, :objective, :eval_metric, :n_estimators, :max_depth) 
                RETURNING hyperparameter_id
            """),
            {
                "hash": param_hash,
                "objective": params['objective'],
                "eval_metric": params['eval_metric'],
                "n_estimators": params['n_estimators'],
                "max_depth": params['max_depth']
            }
        )
        return result.fetchone()[0]

    def _get_or_create_preprocessing_id(self, params: Dict, connection) -> int:
        """Get existing preprocessing_id or create new entry"""
        param_str = f"{params['sampling_method']}_{params['batch_size']}_{params['num_workers']}"
        param_hash = hashlib.sha256(param_str.encode()).hexdigest()

        result = connection.execute(
            text("SELECT preprocessing_id FROM dim_preprocessing WHERE preprocessing_hash = :hash"),
            {"hash": param_hash}
        )
        row = result.fetchone()
        if row:
            return row[0]

        result = connection.execute(
            text("""
                INSERT INTO dim_preprocessing 
                (preprocessing_hash, sampling_method, batch_size, num_workers) 
                VALUES (:hash, :sampling_method, :batch_size, :num_workers) 
                RETURNING preprocessing_id
            """),
            {
                "hash": param_hash,
                "sampling_method": params['sampling_method'],
                "batch_size": params['batch_size'],
                "num_workers": params['num_workers']
            }
        )
        return result.fetchone()[0]

    def _get_or_create_dataset_id(self, params: Dict, connection) -> int:
        """Get existing dataset_id or create new entry"""
        param_str = f"{params['train_samples']}_{params['valid_samples']}_{params['test_samples']}"
        param_hash = hashlib.sha256(param_str.encode()).hexdigest()

        result = connection.execute(
            text("SELECT dataset_id FROM dim_dataset WHERE dataset_hash = :hash"),
            {"hash": param_hash}
        )
        row = result.fetchone()
        if row:
            return row[0]

        result = connection.execute(
            text("""
                INSERT INTO dim_dataset 
                (dataset_hash, train_samples, valid_samples, test_samples) 
                VALUES (:hash, :train_samples, :valid_samples, :test_samples) 
                RETURNING dataset_id
            """),
            {
                "hash": param_hash,
                "train_samples": params['train_samples'],
                "valid_samples": params['valid_samples'],
                "test_samples": params['test_samples']
            }
        )
        return result.fetchone()[0]

    def save_training_results(self, results_df: pd.DataFrame) -> bool:
        """
        Save training results to star schema, populating dimension and fact tables.

        Args:
            results_df (DataFrame): DataFrame with columns matching the original flat schema

        Returns:
            bool: True if save was successful, False otherwise
        """
        if not self.is_connected():
            print("No database connection. Cannot save training results.")
            return False

        try:
            with self.engine.connect() as connection:
                for idx, row in results_df.iterrows():
                    # Get or create dimension IDs
                    model_id = self._get_or_create_model_id(row['model_name'], connection)

                    hyperparameter_id = self._get_or_create_hyperparameter_id({
                        'objective': row['objective'],
                        'eval_metric': row['eval_metric'],
                        'n_estimators': row['n_estimators'],
                        'max_depth': row['max_depth']
                    }, connection)

                    preprocessing_id = self._get_or_create_preprocessing_id({
                        'sampling_method': row['sampling_method'],
                        'batch_size': row['batch_size'],
                        'num_workers': row['num_workers']
                    }, connection)

                    dataset_id = self._get_or_create_dataset_id({
                        'train_samples': row['train_samples'],
                        'valid_samples': row['valid_samples'],
                        'test_samples': row['test_samples']
                    }, connection)

                    # Insert into fact table (with conflict handling)
                    connection.execute(
                        text("""
                            INSERT INTO fact_training_results 
                            (param_id, model_id, hyperparameter_id, preprocessing_id, dataset_id, 
                             test_accuracy, valid_accuracy, training_time)
                            VALUES (:param_id, :model_id, :hyperparameter_id, :preprocessing_id, :dataset_id,
                                    :test_accuracy, :valid_accuracy, :training_time)
                            ON CONFLICT (param_id) DO NOTHING
                        """),
                        {
                            "param_id": idx,
                            "model_id": model_id,
                            "hyperparameter_id": hyperparameter_id,
                            "preprocessing_id": preprocessing_id,
                            "dataset_id": dataset_id,
                            "test_accuracy": row['test_accuracy'],
                            "valid_accuracy": row['valid_accuracy'],
                            "training_time": row['training_time']
                        }
                    )

                connection.commit()
                print(f"Training results saved to star schema (duplicates ignored).")
                return True

        except Exception as e:
            print(f"Failed to save training results: {e}")
            return False

    def get_best_models(self, test_accuracy_threshold: float = None,
                       valid_accuracy_threshold: float = None,
                       training_time_limit: float = None) -> pd.DataFrame:
        """
        Retrieve models that meet specified performance criteria using star schema.

        Args:
            test_accuracy_threshold (float): Minimum test accuracy
            valid_accuracy_threshold (float): Minimum validation accuracy
            training_time_limit (float): Maximum training time in seconds

        Returns:
            pd.DataFrame: DataFrame of models meeting the criteria
        """
        if not self.is_connected():
            print("No database connection. Cannot query model results.")
            return None

        try:
            query = """
            SELECT 
                f.param_id,
                m.model_name,
                h.objective,
                h.eval_metric,
                h.n_estimators,
                h.max_depth,
                p.sampling_method,
                p.batch_size,
                p.num_workers,
                d.train_samples,
                d.valid_samples,
                d.test_samples,
                f.test_accuracy,
                f.valid_accuracy,
                f.training_time,
                f.created_at
            FROM fact_training_results f
            JOIN dim_model m ON f.model_id = m.model_id
            JOIN dim_hyperparameters h ON f.hyperparameter_id = h.hyperparameter_id
            JOIN dim_preprocessing p ON f.preprocessing_id = p.preprocessing_id
            JOIN dim_dataset d ON f.dataset_id = d.dataset_id
            WHERE 1=1
            """

            if test_accuracy_threshold is not None:
                query += f" AND f.test_accuracy >= {test_accuracy_threshold}"
            if valid_accuracy_threshold is not None:
                query += f" AND f.valid_accuracy >= {valid_accuracy_threshold}"
            if training_time_limit is not None:
                query += f" AND f.training_time <= {training_time_limit}"

            order_clauses = []
            if test_accuracy_threshold is not None:
                order_clauses.append("f.test_accuracy DESC")
            if valid_accuracy_threshold is not None:
                order_clauses.append("f.valid_accuracy DESC")
            if training_time_limit is not None:
                order_clauses.append("f.training_time ASC")

            if order_clauses:
                query += " ORDER BY " + ", ".join(order_clauses)

            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                print(f"Retrieved {len(df)} models meeting criteria.")
                return df

        except Exception as e:
            print(f"Failed to query model results: {e}")
            return None

    def get_model_history(self, model_name: str = None) -> pd.DataFrame:
        """
        Retrieve the training history of a specific model using star schema.

        Args:
            model_name (str): Name of the model to query

        Returns:
            pd.DataFrame: DataFrame of model training history
        """
        if not self.is_connected():
            print("No database connection. Cannot query model results.")
            return None

        try:
            if not model_name:
                print("Model name required but not provided.")
                return None

            query = """
            SELECT 
                f.param_id,
                m.model_name,
                h.objective,
                h.eval_metric,
                h.n_estimators,
                h.max_depth,
                p.sampling_method,
                p.batch_size,
                p.num_workers,
                d.train_samples,
                d.valid_samples,
                d.test_samples,
                f.test_accuracy,
                f.valid_accuracy,
                f.training_time,
                f.created_at
            FROM fact_training_results f
            JOIN dim_model m ON f.model_id = m.model_id
            JOIN dim_hyperparameters h ON f.hyperparameter_id = h.hyperparameter_id
            JOIN dim_preprocessing p ON f.preprocessing_id = p.preprocessing_id
            JOIN dim_dataset d ON f.dataset_id = d.dataset_id
            WHERE m.model_name = :model_name
            ORDER BY f.created_at DESC
            """

            df = pd.read_sql(query, self.engine, params={"model_name": model_name})
            return df

        except Exception as e:
            print(f"Failed to query model history: {e}")
            return None

    def get_summary_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get summary statistics using the star schema.

        Returns:
            dict: Summary statistics including total models, average accuracies, etc.
        """
        if not self.is_connected():
            print("No database connection. Cannot query model results.")
            return None

        try:
            with self.engine.connect() as connection:
                # Basic counts
                result = connection.execute(text("SELECT COUNT(*) FROM fact_training_results"))
                total_experiments = result.fetchone()[0]

                # Best accuracies
                result = connection.execute(text("SELECT MAX(test_accuracy) FROM fact_training_results"))
                best_test_acc = result.fetchone()[0]

                result = connection.execute(text("SELECT MAX(valid_accuracy) FROM fact_training_results"))
                best_valid_acc = result.fetchone()[0]

                # Average training time
                result = connection.execute(text("SELECT AVG(training_time) FROM fact_training_results"))
                avg_training_time = result.fetchone()[0]

                # Dimension table counts
                result = connection.execute(text("SELECT COUNT(*) FROM dim_model"))
                total_models = result.fetchone()[0]

                result = connection.execute(text("SELECT COUNT(*) FROM dim_hyperparameters"))
                total_hyperparameter_configs = result.fetchone()[0]

                return {
                    'total_experiments': total_experiments,
                    'total_unique_models': total_models,
                    'total_hyperparameter_configs': total_hyperparameter_configs,
                    'best_test_accuracy': float(best_test_acc) if best_test_acc else None,
                    'best_valid_accuracy': float(best_valid_acc) if best_valid_acc else None,
                    'avg_training_time': float(avg_training_time) if avg_training_time else None
                }

        except Exception as e:
            print(f"Failed to get summary statistics: {e}")
            return None


if __name__ == "__main__":
    # Test the star schema database utility functions
    db = StarSchemaDB()

    if db.is_connected():
        print("Database connection successful.")

        # Example: Get summary statistics
        stats = db.get_summary_stats()
        print("\nDatabase Summary Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        pd.options.display.max_columns = None
        pd.options.display.max_rows = 5
        pd.options.display.width = 1000

        # Example: Query best models
        best_models = db.get_best_models(
            test_accuracy_threshold=0.85,
            valid_accuracy_threshold=0.8,
            training_time_limit=300
        )
        if best_models is not None and not best_models.empty:
            print("\nBest Models:")
            display(best_models)

        # Example: Get model history
        model_history = db.get_model_history(model_name='ResNet50')
        if model_history is not None and not model_history.empty:
            print("\nModel Training History for ResNet50:")
            display(model_history)
    else:
        print("Database connection failed.")