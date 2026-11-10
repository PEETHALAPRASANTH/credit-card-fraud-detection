from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT_DIR / "dataset" / "creditcard.csv"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
METRICS_PATH = REPORT_DIR / "metrics.json"
METADATA_PATH = MODEL_DIR / "metadata.json"

TARGET = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Kaggle dataset has Time, Amount, V1..V28 and Class.
FEATURE_COLUMNS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
