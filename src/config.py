from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "transactions.csv"
MODEL_PATH = ROOT / "models" / "fraud_pipeline.joblib"
META_PATH = ROOT / "models" / "metadata.json"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

TARGET = "is_fraud"
TIME_COL = "transaction_time"
SEED = 42

# temporal split: train < VAL_START <= validation < TEST_START <= test
VAL_START = "2024-08-01"
TEST_START = "2024-09-01"

# business assumptions for the cost metric (adjustable)
REVIEW_COST = 5.0        # cost of manually reviewing one flagged transaction
MISSED_FRAUD_RATE = 1.0  # share of the amount lost when a fraud is not flagged

RAW_COLUMNS = [
    "account_age_days", "total_transactions_user", "avg_amount_user", "amount",
    "country", "bin_country", "channel", "merchant_category", "promo_used",
    "avs_match", "cvv_result", "three_ds_flag", "shipping_distance_km",
    "transaction_time",
]
