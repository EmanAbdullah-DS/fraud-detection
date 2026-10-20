import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from data import load_data, temporal_split
from features import add_history_features
from scoring import load_artifacts, score_transaction


def test_split_is_chronological():
    train, val, test = temporal_split(load_data())
    assert train["transaction_time"].max() < val["transaction_time"].min()
    assert val["transaction_time"].max() < test["transaction_time"].min()


def test_history_features_use_only_the_past():
    df = load_data().head(5000)
    out = add_history_features(df)
    first = out.groupby("user_id").head(1)
    assert (first["prev_txn_count"] == 0).all()
    assert first["hours_since_prev"].isna().all()
    assert first["amount_vs_past_mean"].isna().all()


def test_saved_pipeline_scores_raw_rows():
    pipe, meta = load_artifacts()
    row = dict(meta["reference_row"])
    row["transaction_time"] = "2024-10-15T14:00:00Z"
    res = score_transaction(pipe, meta, row)
    assert 0 <= res["probability"] <= 1
    assert abs(res["contributions"]["effect"].sum()) < 1
    assert res["label"] in ("Low Risk", "High Risk")
