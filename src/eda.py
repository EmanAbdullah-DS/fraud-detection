import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES, TARGET
from data import load_data


def save(name):
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=130)
    plt.close()


def rate_plot(ax, series, title):
    series.plot(kind="bar", ax=ax, color="#2a6f97")
    ax.set_title(title)
    ax.set_ylabel("fraud rate")
    ax.tick_params(axis="x", rotation=30)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_data()
    df["amount_to_avg"] = df["amount"] / df["avg_amount_user"]
    df["country_mismatch"] = (df["country"] != df["bin_country"]).astype(int)
    df["hour"] = df["transaction_time"].dt.hour

    print("rows:", len(df), "| users:", df["user_id"].nunique())
    print("period:", df["transaction_time"].min(), "->", df["transaction_time"].max())
    print("fraud rate: %.4f" % df[TARGET].mean())
    print("missing values:", int(df.isna().sum().sum()), "| duplicate ids:", int(df["transaction_id"].duplicated().sum()))
    users_with_fraud = df.groupby("user_id")[TARGET].max().mean()
    print("users with at least one fraud: %.3f" % users_with_fraud)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    df[TARGET].value_counts().plot(kind="bar", ax=ax[0], color=["#2a6f97", "#c1121f"])
    ax[0].set_title("Class balance")
    ax[0].tick_params(axis="x", rotation=0)
    m = df.groupby(df["transaction_time"].dt.strftime("%Y-%m"))[TARGET].mean()
    m.plot(marker="o", ax=ax[1], color="#2a6f97")
    ax[1].set_title("Monthly fraud rate")
    ax[1].set_ylim(0, 0.04)
    save("01_target.png")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for k, c in [(0, "#2a6f97"), (1, "#c1121f")]:
        ax[0].hist(np.log10(df.loc[df[TARGET] == k, "amount"]), bins=60, alpha=0.6, density=True, color=c, label=f"fraud={k}")
    ax[0].set_title("log10(amount)")
    ax[0].legend()
    for k, c in [(0, "#2a6f97"), (1, "#c1121f")]:
        ax[1].hist(np.log10(df.loc[df[TARGET] == k, "amount_to_avg"]), bins=60, alpha=0.6, density=True, color=c, label=f"fraud={k}")
    ax[1].set_title("log10(amount / user average)")
    save("02_amount_distributions.png")

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    cats = ["channel", "merchant_category", "avs_match", "cvv_result", "three_ds_flag", "promo_used", "country_mismatch"]
    for ax, c in zip(axes.flat, cats):
        rate_plot(ax, df.groupby(c)[TARGET].mean(), c)
    b = pd.qcut(df["account_age_days"], 5)
    rate_plot(axes.flat[7], df.groupby(b, observed=True)[TARGET].mean().rename(lambda x: f"{int(x.left)}-{int(x.right)}"), "account_age_days")
    save("03_fraud_rate_by_category.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, (c, q) in zip(axes, [("shipping_distance_km", 10), ("amount_to_avg", 10), ("hour", 24)]):
        b = pd.qcut(df[c], q, duplicates="drop") if c != "hour" else df[c]
        r = df.groupby(b, observed=True)[TARGET].mean()
        r.index = [round(float(i.mid), 2) if hasattr(i, "mid") else i for i in r.index]
        r.plot(marker="o", ax=ax, color="#2a6f97")
        ax.set_title(f"fraud rate by {c}")
    save("04_fraud_rate_by_numeric.png")

    num_cols = ["account_age_days", "total_transactions_user", "avg_amount_user", "amount", "amount_to_avg",
                "shipping_distance_km", "avs_match", "cvv_result", "three_ds_flag", "promo_used",
                "country_mismatch", TARGET]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)), num_cols, rotation=60, ha="right")
    ax.set_yticks(range(len(num_cols)), num_cols)
    fig.colorbar(im)
    ax.set_title("Correlation")
    save("05_correlation.png")
    print("corr with target:\n", corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).round(3))


if __name__ == "__main__":
    main()
