from dotenv import load_dotenv
import os

load_dotenv("model_version.env", override=False)  # loads variables from model_version.env ONLY if they are not already set in the environment (e.g. by Docker)

MODEL_VERSION = int(os.getenv("MODEL_VERSION", "1"))
MODEL_DIR = os.getenv("MODEL_DIR", f"models/model_v{MODEL_VERSION}")
