import os
from dotenv import load_dotenv
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

# Database configuration
load_dotenv()

host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
database = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")

def create_database_connection():
    """ Create a SQLAlchemy engine for database operations """
    try:
        connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        engine = create_engine(connection_string)

        # Test the connection
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        print("Database connection established")
        return engine

    except Exception as e:
        print(f"Failed to connect to database: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure PostgreSQL is running: brew services start postgresql@14")
        print("2. Check if database exists: psql -d model_results_db -U model_user")
        print("3. Verify credentials in DB_CONFIG")
        return None

def create_model_results_table(engine):
    """ Create the model results table if it doesn't exist """
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
        with engine.connect() as connection:
            connection.execute(text(create_table_sql))
            connection.commit()
        print("Model results table created/verified")
        return True
    except Exception as e:
        print(f"Failed to create model results table: {e}")
        return False

def load_csv_data(csv_file_path):
    """ Load data from CSV file """
    if not Path(csv_file_path).exists():
        print(f"CSV file not found at: {csv_file_path}")
        return None

    try:
        df = pd.read_csv(csv_file_path, index_col='param_id')
        print(f"{len(df)} rows loaded from {csv_file_path}")
        return df
    except Exception as e:
        print(f"Failed to load data from {csv_file_path}: {e}")
        return None

def upload_data_to_postgres(df, engine, table_name='model_results'):
    """ Upload dataframe to PostgreSQL database """
    try:
        # Upload data
        rows_affected = df.to_sql(table_name, engine, if_exists='replace', index=True, index_label='param_id')
        print(f"Successfully uploaded {len(df)} rows to {table_name}")
        return True
    except Exception as e:
        print(f"Failed to upload data: {e}")
        return False

def verify_upload(engine, table_name='model_results'):
    """ Verify upload of data by querying the database """
    try:
        with engine.connect() as connection:
            # Count total rows
            result = connection.execute(text(f"SELECT * FROM {table_name}"))
            count = result.rowcount
            print(f"Database now contains {count} rows")

            # Show latest entries
            result = connection.execute(text(f"""
                SELECT param_id, model_name, test_accuracy, valid_accuracy
                FROM {table_name}
                ORDER BY param_id DESC
                LIMIT 5
                """))

            print("\nLatest 5 entries:")
            print("      param_id       | model_name | test_acc | valid_acc")
            print("-"*60)
            for row in result:
                print(f"{row[0][:20]} | {row[1]} | {row[2]:.4f} | {row[3]:.4f}")
        return True
    except Exception as e:
        print(f"Failed to verify upload: {e}")
        return False

def main():
    """ Main function to orchestrate migration """
    print("Starting CSV to PostgreSQL migration")

    # CSV file path
    csv_file_path = 'model_training_results.csv'

    # Step 1: Load CSV data
    df = load_csv_data(csv_file_path)
    if df is None:
        return False

    # Step 2: Create database connection
    engine = create_database_connection()
    if engine is None:
        return False

    # Step 3: Create table
    if not create_model_results_table(engine):
        return False

    # Step 4: Upload data
    if not upload_data_to_postgres(df, engine, table_name='model_results'):
        return False

    # Step 5: Verify upload
    if not verify_upload(engine, table_name='model_results'):
        return False

    print("Finished migration successfully")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)