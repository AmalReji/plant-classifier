from dotenv import load_dotenv
import os

load_dotenv()  # loads variables from .env ONLY if they are not already set in the environment (e.g. by Docker)

MODEL_DIR = os.getenv("MODEL_DIR", "/app/models/model_v1")
MODEL_VERSION = int(os.getenv("MODEL_VERSION", "1"))