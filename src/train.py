import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from config import (FIGURES, META_PATH, MODEL_PATH, RAW_COLUMNS, REPORTS, REVIEW_COST,
                    SEED, TARGET)
from cost import best_threshold, total_cost
from data import load_data, temporal_split
from explain import reference_row
from features import (HISTORY, MODEL_FEATURES, add_history_features,
                      add_transaction_features)
from pipeline import boosting_pipeline, forest_pipeline, logistic_pipeline


def scores(y, p):
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p)}


def at_threshold(y, amount, p, t):
    flag = p >= t
    return {
        "threshold": float(t),
        "precision": precision_score(y, flag),
        "recall": recall_score(y, flag),
        "f1": f1_score(y, flag),
        "flag_rate": float(flag.mean()),
        "cost": total_cost(y, amount, flag),
    }


def history_ablation(train, val, test):
    """Does adding past-only rolling/user-context features help?"""
    full = add_history_features(pd.concat([train, val, test]).sort_values(["transaction_time", "transaction_id"]))
    parts = {k: full.loc[d.index] for k, d in [("train", train), ("test", test)]}
    res = {}
    for name, extra in [("transaction features", []), ("+ history features", HISTORY)]:
        X = []
        for k in ("train", "test"):
            f = add_transaction_features(parts[k])
            for c in extra:
                f[c] = parts[k][c].values
            f["channel"] = f["channel"].astype("category")
            f["merchant_category"] = f["merchant_category"].astype("category")
            X.append(f)
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=40,
                                           categorical_features="from_dtype", random_state=SEED)
        m.fit(X[0], parts["train"][TARGET])
        res[name] = scores(parts["test"][TARGET], m.predict_proba(X[1])[:, 1])
    return res


def unseen_user_check(train, test, pipe_factory):
    rng = np.random.default_rng(SEED)
    users = train["user_id"].unique()
    held = set(rng.choice(users, size=len(users) // 5, replace=False))
    tr = train[~train["user_id"].isin(held)]
    te = test[test["user_id"].isin(held)]
    m = pipe_factory().fit(tr[RAW_COLUMNS], tr[TARGET])
    return scores(te[TARGET], m.predict_proba(te[RAW_COLUMNS])[:, 1])


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_data()
    train, val, test = temporal_split(df)
    print("train %d | val %d | test %d" % (len(train), len(val), len(test)))
    Xtr, ytr = train[RAW_COLUMNS], train[TARGET]
    Xva, yva = val[RAW_COLUMNS], val[TARGET]
    Xte, yte = test[RAW_COLUMNS], test[TARGET]

    candidates = {
        "Logistic Regression": logistic_pipeline(),
        "Random Forest": forest_pipeline(),
        "Gradient Boosting": boosting_pipeline(),
    }
    results, fitted = {}, {}
    for name, pipe in candidates.items():
        pipe.fit(Xtr, ytr)
        fitted[name] = pipe
        results[name] = scores(yva, pipe.predict_proba(Xva)[:, 1])
        print(name, {k: round(v, 4) for k, v in results[name].items()})

    grid = [dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40),
            dict(max_iter=500, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=60, l2_regularization=1.0),
            dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=100, l2_regularization=1.0),
            dict(max_iter=600, learning_rate=0.02, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=2.0)]
    best_name, best_score = "Gradient Boosting", results["Gradient Boosting"]["pr_auc"]
    for i, params in enumerate(grid):
        name = f"Gradient Boosting (tuned {i + 1})"
        pipe = boosting_pipeline(**params).fit(Xtr, ytr)
        fitted[name] = pipe
        results[name] = scores(yva, pipe.predict_proba(Xva)[:, 1])
        print(name, params, {k: round(v, 4) for k, v in results[name].items()})
        if results[name]["pr_auc"] > best_score:
            best_name, best_score = name, results[name]["pr_auc"]
    print("selected on validation:", best_name)
    final = fitted[best_name]

    p_val = final.predict_proba(Xva)[:, 1]
    p_test = final.predict_proba(Xte)[:, 1]
    thr, _ = best_threshold(yva.values, val["amount"].values, p_val)

    pr, rc, th = precision_recall_curve(yva, p_val)
    f1s = 2 * pr * rc / np.clip(pr + rc, 1e-9, None)
    f1_thr = float(th[np.argmax(f1s[:-1])])

    test_rows = {}
    for name, pipe in fitted.items():
        if name in ("Logistic Regression", "Random Forest", "Gradient Boosting") or name == best_name:
            p = pipe.predict_proba(Xte)[:, 1]
            t_m, _ = best_threshold(yva.values, val["amount"].values, pipe.predict_proba(Xva)[:, 1])
            test_rows[name] = {**scores(yte, p), **at_threshold(yte.values, test["amount"].values, p, t_m)}
    test_rows["__f1_threshold"] = at_threshold(yte.values, test["amount"].values, p_test, f1_thr)

    no_model = total_cost(yte.values, test["amount"].values, np.zeros(len(test)))
    flag_all = total_cost(yte.values, test["amount"].values, np.ones(len(test)))
    ranked = np.argsort(-p_test)
    k = int(0.01 * len(test))
    recall_top1 = yte.values[ranked[:k]].sum() / yte.sum()

    mae = float(np.mean(np.abs(yte.values - p_test)))
    rmse = float(np.sqrt(np.mean((yte.values - p_test) ** 2)))

    ablation = history_ablation(train, val, test)
    unseen = unseen_user_check(train, test, lambda: boosting_pipeline(**(grid[int(best_name[-2]) - 1] if "tuned" in best_name else {})))

    imp_idx = np.random.default_rng(SEED).choice(len(val), size=min(20000, len(val)), replace=False)
    imp = permutation_importance(final, Xva.iloc[imp_idx], yva.iloc[imp_idx], scoring="average_precision",
                                 n_repeats=3, random_state=SEED, n_jobs=-1)
    imp_df = pd.DataFrame({"field": RAW_COLUMNS, "importance": imp.importances_mean}).sort_values("importance")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp_df["field"], imp_df["importance"], color="#2a6f97")
    ax.set_title("Permutation importance (PR-AUC drop, validation)")
    plt.tight_layout(); plt.savefig(FIGURES / "06_feature_importance.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    pr_t, rc_t, _ = precision_recall_curve(yte, p_test)
    ax[0].plot(rc_t, pr_t, color="#2a6f97")
    ax[0].set(xlabel="recall", ylabel="precision", title="Precision-Recall (test)")
    grid_t = np.linspace(0.02, 0.9, 120)
    costs = [total_cost(yte.values, test["amount"].values, p_test >= t) for t in grid_t]
    ax[1].plot(grid_t, costs, color="#2a6f97")
    ax[1].axvline(thr, color="#c1121f", ls="--", label=f"chosen threshold {thr:.2f}")
    ax[1].set(xlabel="threshold", ylabel="total cost", title="Cost vs threshold (test)")
    ax[1].legend()
    plt.tight_layout(); plt.savefig(FIGURES / "07_pr_and_cost.png", dpi=130); plt.close()

    chosen = test_rows[best_name]
    summary = {
        "selected_model": best_name,
        "validation": results,
        "test": test_rows,
        "cost_baselines_test": {"no_model": no_model, "flag_everything": flag_all,
                                "model_at_cost_threshold": chosen["cost"],
                                "saving_vs_no_model_pct": 100 * (1 - chosen["cost"] / no_model)},
        "recall_in_top_1pct_test": float(recall_top1),
        "probability_mae_rmse_test": {"mae": mae, "rmse": rmse},
        "history_feature_ablation_test": ablation,
        "unseen_users_check_test": unseen,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))

    rows = [{"model": n, **{k: round(v, 4) for k, v in r.items() if k in ("pr_auc", "roc_auc", "precision", "recall", "f1", "cost")}}
            for n, r in test_rows.items() if not n.startswith("__")]
    pd.DataFrame(rows).to_csv(REPORTS / "test_comparison.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("validation", "test")}, indent=2, default=float))

    meta = {
        "model": best_name,
        "threshold": thr,
        "review_cost": REVIEW_COST,
        "val_start": "2024-08-01",
        "test_start": "2024-09-01",
        "categories": {"channel": sorted(df["channel"].unique().tolist()),
                       "merchant_category": sorted(df["merchant_category"].unique().tolist()),
                       "country": sorted(df["country"].unique().tolist())},
        "reference_row": reference_row(train),
        "ranges": {c: [float(df[c].min()), float(df[c].max()), float(df[c].median())]
                   for c in ["account_age_days", "total_transactions_user", "avg_amount_user", "amount", "shipping_distance_km"]},
        "feature_importance": imp_df.sort_values("importance", ascending=False).set_index("field")["importance"].round(5).to_dict(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    joblib.dump(final, MODEL_PATH)
    print("saved", MODEL_PATH)


if __name__ == "__main__":
    main()
