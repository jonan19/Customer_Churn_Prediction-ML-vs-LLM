"""Central configuration -- all constants live here, nowhere else."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = ROOT_DIR / "data" / "raw" / "telco_churn_raw.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for d in (PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Fallback source if the committed raw CSV is ever missing. This is the
# standard IBM-published "Telco Customer Churn" sample dataset, mirrored
# on GitHub; the raw file is also committed to this repo (data/raw/) so
# the pipeline reproduces identically even if this URL ever changes.
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)

ID_COL = "customerID"
TARGET_COL = "Churn"

TEST_SIZE = 0.20          # held out, touched exactly once
N_CV_FOLDS = 5
RANDOM_STATE = 42

# Add-on service columns used to engineer `num_services`.
SERVICE_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

TENURE_BUCKET_EDGES = [-1, 12, 24, 48, 60, 1000]
TENURE_BUCKET_LABELS = ["0-12mo", "13-24mo", "25-48mo", "49-60mo", "61mo+"]
