import os
import sys
from pathlib import Path

def test_model_loads():
    original = os.getcwd()

    try:
        app_dir = Path(__file__).resolve().parents[1] / "app"
        os.chdir(app_dir)
        sys.path.insert(0, str(app_dir))

        from predict import load_model

        model = load_model()
        assert model is not None, "Model should be loaded successfully"
    finally:
        os.chdir(original)
        sys.path.pop(0)