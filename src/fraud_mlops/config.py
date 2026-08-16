from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "creditcard.csv"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODEL_DIR / "fraud_model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"
REFERENCE_STATS_PATH = REPORT_DIR / "reference_stats.json"

FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
TARGET_COLUMN = "Class"
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.5
MODEL_NAME = "credit-card-fraud-detector"
