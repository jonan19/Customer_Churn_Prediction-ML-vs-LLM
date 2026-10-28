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

# ---- LLM cost/accuracy comparison (add-on) --------------------------------
# llama-3.1-8b-instant moved to Groq's enterprise/contact-sales tier as of
# Oct 2026; openai/gpt-oss-20b is the current cheapest self-serve,
# production (non-preview) model on GroqCloud. Pricing verified against
# console.groq.com/docs/models at the time this was written -- re-check
# before relying on this number for a real budget decision.
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL = "openai/gpt-oss-20b"
LLM_PRICE_PER_1M_INPUT_TOKENS = 0.075
LLM_PRICE_PER_1M_OUTPUT_TOKENS = 0.30

LLM_SAMPLE_SIZE = 200       # stratified subsample of the test set, to bound API cost/time
LLM_FEWSHOT_PER_CLASS = 3   # few-shot examples per class, drawn from TRAIN only

# openai/gpt-oss-20b is a REASONING model: most of its completion tokens
# go to a hidden chain-of-thought, not the visible answer (measured: ~60%
# of output tokens at "medium" effort, the API default). "low" keeps the
# comparison's cost/latency realistic for a task this simple; "high" was
# tested and burned a 300-token budget on reasoning alone without ever
# emitting a final answer -- itself a finding worth reporting.
LLM_REASONING_EFFORT = "low"
LLM_MAX_TOKENS = 150
