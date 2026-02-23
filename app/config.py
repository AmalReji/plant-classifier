from dotenv import load_dotenv
import os

load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_VERSION = int(os.getenv("MODEL_VERSION"))