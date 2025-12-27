"""
Database utility functions for model training results.
This module can be imported for direct database operations.
"""
import os
from typing import Dict, Any, Optional

import pandas as pd
from IPython.core.display_functions import display
from dotenv import load_dotenv
from pandas.core.interchange.dataframe_protocol import DataFrame
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert


class ModelResultsDB:
    """ Class to handle database operations for model training results """

    def __init__(self, db_config: Dict[str, Any]= None):
        """ Initialise the database connection """
        if db_config is None:
            # Default configuration using .env
            load_dotenv()  # allow os.getenv() to access variables in .env file

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
        self._create_model_results_table()  # Check model_results table exists, and create if not

    def _connect(self):
        """ Create a SQLAlchemy engine for database operations """
        try:
            connection_string = f"postgresql://{self.db_config['user']}:{self.db_config['password']}@" \
                                f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
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
        """ Check if the database connection is established """
        return self.engine is not None

    def _create_model_results_table(self):
        """ Create the model results table if it doesn't exist """
        if not self.is_connected():
            print("No database connection. Cannot create table.")
            return

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS model_results (
            param_id VARCHAR(100) PRIMARY KEY,
            sampling_method VARCHAR(50),
            objective VARCHAR(50),
            num_workers INTEGER,
            n_estimators INTEGER,
            model_name VARCHAR(100),
            max_depth INTEGER,
            eval_metric VARCHAR(50),
            batch_size INTEGER,
            test_accuracy FLOAT,
            valid_accuracy FLOAT,
            training_time FLOAT,
            train_samples INTEGER,
            test_samples INTEGER,
            valid_samples INTEGER
            );
            """

        try:
            with self.engine.connect() as connection:
                connection.execute(text(create_table_sql))
                connection.commit()
                print("Model results table is ready.")
        except Exception as e:
            print(f"Failed to create table: {e}")

    def save_model_results(self, model_result: DataFrame):
        """ Save a DataFrame of model results to the database, ignoring duplicates (rows already in the table)

        Args:
            model_result (DataFrame): List of model hyperparameters and results

        Returns:
            bool: True if save was successful, False otherwise
        """
        if not self.is_connected():
            print("No database connection. Cannot save model results.")
            return False

        try:
            # Custom UPSERT function with ON CONFLICT DO NOTHING
            def upsert(table, conn, keys, data_iter):
                data = [dict(zip(keys, row)) for row in data_iter]
                stmt = insert(table.table).values(data)
                stmt = stmt.on_conflict_do_nothing(index_elements=['param_id'])
                conn.execute(stmt)

            model_result.to_sql("model_results", self.engine, if_exists='append',
                                index_label="param_id",  # don't insert pandas index
                                method=upsert  # use custom upsert function
                                )


            # # Write to a temporary table and then upsert to avoid duplicates
            # temp_table = 'temp_model_results'
            # model_result.to_sql(temp_table, self.engine, if_exists='replace', index_label='param_id')
            # with self.engine.connect() as connection:
            #     connection.execute(
            #         text(f"""
            #         INSERT INTO model_results
            #         SELECT * FROM {temp_table}
            #         ON CONFLICT (param_id) DO NOTHING;
            #         DROP TABLE {temp_table};
            #         """)
            #     )
            #     connection.commit()
            print("Model results saved to database (duplicates ignored).")
            return True
        except Exception as e:
            print(f"Failed to save model results: {e}")
            return False

    def get_best_models(self, test_accuracy_threshold: float = None, valid_accuracy_threshold: float = None, training_time_limit: float = None) -> pd.DataFrame:
        """ Retrieve models that meet specified performance criteria

        Args:
            test_accuracy_threshold (float): Minimum test accuracy
            valid_accuracy_threshold (float): Minimum validation accuracy
            training_time_limit (float): Maximum training time in seconds

        Returns:
            pd.DataFrame: DataFrame of models meeting the criteria, or None if no connection
        """
        if not self.is_connected():
            print("No database connection. Cannot query model results.")
            return None
        try:
            query = "SELECT * FROM model_results WHERE 1=1"
            if test_accuracy_threshold is not None:
                query += f" AND test_accuracy >= {test_accuracy_threshold}"
            if valid_accuracy_threshold is not None:
                query += f" AND valid_accuracy >= {valid_accuracy_threshold}"
            if training_time_limit is not None:
                query += f" AND training_time <= {training_time_limit}"
            order_clauses = []
            if test_accuracy_threshold is not None:
                order_clauses.append("test_accuracy DESC")
            if valid_accuracy_threshold is not None:
                order_clauses.append("valid_accuracy DESC")
            if training_time_limit is not None:
                order_clauses.append("training_time ASC")
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
        """ Retrieve the training history of a specific model

        Args:
            model_name (str): Name of the model to query

        Returns:
            pd.DataFrame: DataFrame of model training history, or None if no connection
        """
        if not self.is_connected():
            print("No database connection. Cannot query model results.")
            return None
        try:
            if model_name:
                query = f"SELECT * FROM model_results WHERE model_name = %s ORDER BY param_id DESC"
                df = pd.read_sql(query, self.engine, params=(model_name,), index_col='param_id')
            else:
                print("Model name required but not provided.")
                return None

            return df

        except Exception as e:
            print(f"Failed to query model history: {e}")
            return None


    def get_summary_stats(self) -> Optional[Dict[str, Any]]:
        """ Get summary statistics of the model results table

        Returns:
            dict: Summary statistics including total models, average accuracies, and average training time
        """
        if not self.is_connected():
            print("No database connection. Cannot query model results.")
            return None
        try:
            with self.engine.connect() as connection:
                # Basic counts
                result = connection.execute(text("SELECT COUNT(*) FROM model_results"))
                total_experiments = result.fetchone()[0]

                # Best accuracy
                result = connection.execute(text("SELECT MAX(test_accuracy) FROM model_results"))
                best_test_acc = result.fetchone()[0]

                result = connection.execute(text("SELECT MAX(valid_accuracy) FROM model_results"))
                best_valid_acc = result.fetchone()[0]

                # Average training time
                result = connection.execute(text("SELECT AVG(training_time) FROM model_results"))
                avg_training_time = result.fetchone()[0]

                return {
                    'total_experiments': total_experiments,
                    'best_test_accuracy': float(best_test_acc) if best_test_acc else None,
                    'best_valid_accuracy': float(best_valid_acc) if best_valid_acc else None,
                    'avg_training_time': float(avg_training_time) if avg_training_time else None
                }
        except Exception as e:
            print(f"Failed to get summary statistics: {e}")
            return None

if __name__ == "__main__":
    # Test the database utility functions
    db = ModelResultsDB()

    if db.is_connected():
        print("Database connection successful.")

        # Example: Get summary statistics
        stats = db.get_summary_stats()
        print("Database Summary Statistics:")
        print(stats)

        pd.options.display.max_columns = None
        pd.options.display.max_rows = 5
        pd.options.display.width = 1000

        # Example: Query best models
        best_models = db.get_best_models(test_accuracy_threshold=0.85, valid_accuracy_threshold=0.8, training_time_limit=300)
        print("\nBest Model History:")
        display(best_models)

        # Example: Get model history
        model_history = db.get_model_history(model_name='ResNet50')
        print("\nModel Training History for ResNet50:")
        display(model_history)

    else:
        print("Database connection failed.")