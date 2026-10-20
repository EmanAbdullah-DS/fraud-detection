# Transaction Fraud Detection

End-to-end fraud scoring project: data understanding, EDA, leak-safe features, baseline and boosted models, cost-aware evaluation, and a Streamlit app that scores a transaction.

## Data
`data/transactions.csv` — 299,695 card transactions from 6,000 users, Jan–Oct 2024, 2.2% fraud (6,612 cases). No missing values, no duplicate ids. About 31% of users have at least one fraud.

## Structure
```
src/config.py     paths, split dates, cost assumptions
src/data.py       loading and chronological split
src/features.py   transaction features + past-only history features
src/pipeline.py   preprocessing + model pipelines (raw columns in, probability out)
src/cost.py       business cost and threshold search
src/explain.py    per-transaction explanation (Shapley over input groups)
src/eda.py        EDA figures
src/train.py      training, comparison, evaluation, saving
src/scoring.py    scoring helper used by the app
app/app.py        Streamlit app
tests/            split, leakage and scoring checks
models/           fraud_pipeline.joblib, metadata.json
reports/          metrics.json, test_comparison.csv, figures/
```

## Run
```
pip install -r requirements.txt
cd src
python eda.py
python train.py
cd ..
python -m pytest -q tests
streamlit run app/app.py
```

## Method
| Stage | What was done |
|---|---|
| Data understanding | Users, time span, user attributes, labels, class balance |
| EDA | Fraud rate by channel, AVS/CVV/3DS, country mismatch, account age, shipping distance, amount relative to the user's average; correlations |
| Preprocessing | Scaling and one-hot for the linear model, ordinal coding for trees; raw country columns are replaced by a mismatch flag |
| Feature engineering | Transaction features (log amount, amount / user average, country mismatch, hour, weekday). Past-only history features (previous count, hours since previous, amount vs past mean, rolling mean of last 3, new country) — each statistic is shifted so a row never sees itself or the future |
| Baselines | Logistic Regression, Random Forest |
| Advanced model | Histogram gradient boosting, small grid chosen on validation PR-AUC |
| Evaluation | PR-AUC, ROC-AUC, precision/recall/F1, probability MAE/RMSE, and a cost metric |
| Deployment | Streamlit app; preprocessing and model are saved as one pipeline |

**Split.** Chronological: train Jan–Jul (209k), validation Aug (30k), test Sep–Oct (60k). Model selection and the threshold use validation only; test is scored once. The same users appear in all periods, as in production; a separate check holds out 20% of users entirely.

**Cost.** A flagged transaction costs 5 (manual review). A missed fraud costs its full amount. Both are assumptions in `src/config.py`. The threshold minimises total cost on validation.

## Results (test set, Sep–Oct)
| Model | PR-AUC | ROC-AUC | Precision | Recall | Total cost |
|---|---|---|---|---|---|
| Logistic Regression | 0.644 | 0.949 | 0.170 | 0.867 | 65,332 |
| Random Forest | 0.851 | 0.972 | 0.669 | 0.827 | 54,665 |
| Gradient Boosting | 0.854 | 0.977 | 0.608 | 0.827 | 56,876 |
| **Gradient Boosting (tuned)** | **0.857** | **0.977** | 0.551 | 0.839 | **52,597** |

Precision, recall and cost are at each model's cost-optimal threshold (the tuned model: 0.052).

- Cost with no model: 771,538. Flagging everything: 300,970. Tuned model: 52,597 (93% lower than no model).
- The F1-optimal threshold (0.48) gives higher precision (0.95) but a higher cost (89,241), because it misses more large frauds. This is the cost-versus-RMSE point: the threshold that looks best on a standard metric is not the cheapest.
- Unseen users (20% held out): PR-AUC 0.878, so the model is not relying on user identity.
- History features: PR-AUC 0.8543 → 0.8557. The gain is negligible, so the deployed model uses only single-transaction features. This also lets the app score a transaction without a user history.
- A sequence model (LSTM/GRU) was not tried: the ablation shows no usable temporal signal beyond what the current row already carries.

Strongest signals: shipping distance, amount relative to the user's average, AVS/CVV/3DS failures, country mismatch, web channel, young accounts (see `reports/figures/`).

## Notes
- XGBoost and LightGBM could not be installed in the build environment, so scikit-learn's `HistGradientBoostingClassifier` (same algorithm family) is used. Swapping in either library only changes `boosting_pipeline` in `src/pipeline.py`.
- The explanation shows each input group's contribution compared with a typical legitimate transaction. Inputs are correlated (for example, a long shipping distance is common when the card country differs), so contributions can offset each other.
- A score is an estimate, not proof of fraud; high-risk cases need human review.
- The Streamlit app was written and checked at the scoring level (`tests/`) but not launched in the build environment, because Streamlit could not be installed there.
