import numpy as np
import pandas as pd

CATEGORICAL = ["channel", "merchant_category"]
NUMERIC = [
    "account_age_days", "total_transactions_user", "avg_amount_user", "amount",
    "log_amount", "amount_to_avg", "country_mismatch", "promo_used", "avs_match",
    "cvv_result", "three_ds_flag", "shipping_distance_km", "hour", "day_of_week",
]
MODEL_FEATURES = NUMERIC + CATEGORICAL

HISTORY = ["prev_txn_count", "hours_since_prev", "amount_vs_past_mean",
           "rolling_mean_3", "new_country_for_user"]


def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features computed from a single transaction row (usable at scoring time)."""
    out = df.copy()
    ts = pd.to_datetime(out["transaction_time"], utc=True, format="ISO8601")
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["log_amount"] = np.log1p(out["amount"])
    out["amount_to_avg"] = out["amount"] / out["avg_amount_user"].clip(lower=1e-6)
    out["country_mismatch"] = (out["country"] != out["bin_country"]).astype(int)
    return out[MODEL_FEATURES]


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Past-only per-user features. Every statistic is shifted so a row never
    sees its own value or any later transaction of the same user."""
    out = df.sort_values(["transaction_time", "transaction_id"]).copy()
    g = out.groupby("user_id")
    out["prev_txn_count"] = g.cumcount()
    out["hours_since_prev"] = g["transaction_time"].diff().dt.total_seconds() / 3600
    past_mean = g["amount"].transform(lambda s: s.shift().expanding().mean())
    out["amount_vs_past_mean"] = out["amount"] / past_mean
    out["rolling_mean_3"] = g["amount"].transform(
        lambda s: s.shift().rolling(3, min_periods=1).mean())
    first_seen = ~out.duplicated(["user_id", "country"])
    out["new_country_for_user"] = (first_seen & (out["prev_txn_count"] > 0)).astype(int)
    return out
