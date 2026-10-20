from itertools import product
from math import factorial

import numpy as np
import pandas as pd

from config import RAW_COLUMNS

# related inputs are explained together (e.g. the two countries define a mismatch)
GROUPS = {
    "Amount vs user average": ["amount", "avg_amount_user"],
    "Country / card country": ["country", "bin_country"],
    "Account profile": ["account_age_days", "total_transactions_user"],
    "Channel": ["channel"],
    "Merchant category": ["merchant_category"],
    "Promo used": ["promo_used"],
    "AVS match": ["avs_match"],
    "CVV result": ["cvv_result"],
    "3-D Secure": ["three_ds_flag"],
    "Shipping distance": ["shipping_distance_km"],
    "Transaction time": ["transaction_time"],
}
EXPLAIN_FIELDS = [f for cols in GROUPS.values() for f in cols]


def reference_row(train_df: pd.DataFrame) -> dict:
    """A typical legitimate transaction: median for numbers, mode for categories."""
    legit = train_df[train_df["is_fraud"] == 0]
    ref = {}
    for c in EXPLAIN_FIELDS:
        if c == "transaction_time":
            ref[c] = str(legit[c].sort_values().iloc[len(legit) // 2])
        elif pd.api.types.is_numeric_dtype(legit[c]):
            ref[c] = float(legit[c].median())
        else:
            ref[c] = legit[c].mode().iloc[0]
    return ref


def field_contributions(pipeline, row: dict, reference: dict) -> pd.DataFrame:
    """Exact Shapley values over input groups. For each coalition the groups outside
    it take the reference (typical legitimate) values. Positive = pushes risk up."""
    names = list(GROUPS)
    n = len(names)
    masks = np.array(list(product([0, 1], repeat=n)), dtype=bool)
    rows = []
    for m in masks:
        r = dict(reference)
        for use, g in zip(m, names):
            if use:
                for f in GROUPS[g]:
                    r[f] = row[f]
        rows.append(r)
    batch = pd.DataFrame(rows)
    batch["transaction_time"] = batch["transaction_time"].astype(str)
    p = pipeline.predict_proba(batch[RAW_COLUMNS])[:, 1]
    value = {tuple(m): v for m, v in zip(masks, p)}
    weight = [factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)]
    out = []
    for i, g in enumerate(names):
        phi = 0.0
        for m in masks:
            if not m[i]:
                with_i = m.copy()
                with_i[i] = True
                phi += weight[int(m.sum())] * (value[tuple(with_i)] - value[tuple(m)])
        out.append((g, phi))
    df = pd.DataFrame(out, columns=["field", "effect"])
    return df.reindex(df["effect"].abs().sort_values(ascending=False).index).reset_index(drop=True)
