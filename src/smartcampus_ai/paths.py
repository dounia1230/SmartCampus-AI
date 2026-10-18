"""Project paths shared by command-line and application modules."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
CLASSIFIER_PATH = MODEL_DIR / "classifier.joblib"
PROCEDURES_PATH = DATA_DIR / "procedures.json"
