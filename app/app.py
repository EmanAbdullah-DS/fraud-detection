import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import datetime as dt

import streamlit as st

from scoring import load_artifacts, score_transaction

st.set_page_config(page_title="Transaction Risk Scoring", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 1200px;}
    .hero {background: linear-gradient(120deg, #0b3954 0%, #087e8b 100%);
padding: 1.6rem 2rem; border-radius: 16px; color: #fff; margin-bottom: 1.2rem;}
    .hero h1 {margin: 0; font-size: 1.9rem; color: #fff;}
    .hero p {margin: .3rem 0 0 0; opacity: .9;}
    .card {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 1.1rem 1.3rem;
        background: rgba(128,128,128,.06); margin-bottom: 1rem;}
    .card h4 {margin: 0 0 .6rem 0;}
    .result {border-radius: 16px; padding: 1.4rem 1.6rem; color: #fff; margin-bottom: 1rem;}
    .result.high {background: linear-gradient(120deg, #9d0208, #dc2f02);}
    .result.low {background: linear-gradient(120deg, #1b7f4b, #2fb36d);}
    .result .label {font-size: .95rem; opacity: .9;}
    .result .value {font-size: 2.6rem; font-weight: 700; line-height: 1.1;}
    .result .tag {font-size: 1.2rem; font-weight: 600; margin-top: .2rem;}
    .meter {position: relative; height: 14px; border-radius: 8px; margin: 1.1rem 0 .3rem 0;
            background: linear-gradient(90deg, #2fb36d 0%, #f4d35e 45%, #dc2f02 100%);}
    .meter .pin {position: absolute; top: -5px; width: 4px; height: 24px; background: #fff;
                border-radius: 2px; box-shadow: 0 0 0 1px rgba(0,0,0,.4);}
    .meter .thr {position: absolute; top: -5px; width: 2px; height: 24px; background: #000; opacity: .7;}
    .scale {display: flex; justify-content: space-between; font-size: .75rem; opacity: .85;}
    .row {display: flex; align-items: center; margin: .35rem 0;}
    .row .name {width: 190px; font-size: .9rem;}
    .row .track {flex: 1; height: 16px; background: rgba(128,128,128,.15); border-radius: 8px; position: relative;}
    .row .bar {position: absolute; top: 0; height: 16px; border-radius: 8px;}
    .row .num {width: 70px; text-align: right; font-size: .85rem; font-variant-numeric: tabular-nums;}
    .legend {font-size: .8rem; opacity: .8; margin-top: .5rem;}
    .note {border-left: 4px solid #087e8b; padding: .6rem 1rem; background: rgba(8,126,139,.08);
        border-radius: 6px; font-size: .9rem;}
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        width: 100%; border-radius: 10px; font-weight: 600; padding: .6rem 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_artifacts():
    return load_artifacts()


pipeline, meta = get_artifacts()
cats = meta["categories"]
ref = meta["reference_row"]
countries = cats["country"]

PRESETS = {
    "Typical legitimate": dict(amount=90.0, avg=90.0, age=900, n_tx=50, dist=250.0, country="FR", bin="FR",
channel="app", cat="grocery", promo=False, avs=True, cvv=True, tds=True, hour=14),
    "Suspicious": dict(amount=900.0, avg=90.0, age=60, n_tx=45, dist=1800.0, country="US", bin="GB",
channel="web", cat="electronics", promo=True, avs=False, cvv=False, tds=False, hour=3),
}


def apply_preset():
    p = PRESETS.get(st.session_state["preset"])
    if p:
        for k, v in p.items():
            st.session_state[f"in_{k}"] = v


def default(key, value):
    st.session_state.setdefault(f"in_{key}", value)


base = PRESETS["Typical legitimate"]
for k, v in base.items():
    default(k, v)
st.session_state["in_country"] = st.session_state["in_country"] if st.session_state["in_country"] in countries else countries[0]
st.session_state["in_bin"] = st.session_state["in_bin"] if st.session_state["in_bin"] in countries else countries[0]
st.markdown(
    """
    <div class="hero">
        <h1>🛡️ Transaction Risk Scoring</h1>
        <p>Estimate the fraud risk of a card transaction and see which inputs drive the score.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    st.selectbox("Load an example", ["Custom"] + list(PRESETS), key="preset", on_change=apply_preset)
    st.divider()
    threshold = st.slider("High Risk threshold", 0.01, 0.99, float(round(meta["threshold"], 2)), 0.01,
        help="Transactions with a probability at or above this value are flagged High Risk.")
    st.caption(
        f"Default {meta['threshold']:.2f} minimises total cost: a review costs "
        f"{meta['review_cost']:.0f} per flagged transaction and a missed fraud costs the transaction amount."
    )

left, right = st.columns([1.05, 1], gap="large")

with left:
    with st.form("tx_form"):
        st.markdown("##### Transaction")
        a, b = st.columns(2)
        amount = a.number_input("Amount", 1.0, 20000.0, step=5.0, key="in_amount")
        distance = b.number_input("Shipping distance (km)", 0.0, 10000.0, step=10.0, key="in_dist")
        c, d = st.columns(2)
        channel = c.selectbox("Channel", cats["channel"], key="in_channel")
        category = d.selectbox("Merchant category", cats["merchant_category"], key="in_cat")
        e, f = st.columns(2)
        country = e.selectbox("Transaction country", countries, key="in_country")
        bin_country = f.selectbox("Card (BIN) country", countries, key="in_bin")
        g, h = st.columns(2)
        date = g.date_input("Date", dt.date(2024, 10, 15))
        hour = h.slider("Hour of day", 0, 23, key="in_hour")

        st.markdown("##### Customer")
        i, j, k = st.columns(3)
        avg_amount = i.number_input("Average amount", 1.0, 5000.0, step=5.0, key="in_avg")
        age = j.number_input("Account age (days)", 1, 3650, key="in_age")
        n_tx = k.number_input("Total transactions", 1, 500, key="in_n_tx")

        st.markdown("##### Verification")
        m1, m2, m3, m4 = st.columns(4)
        avs = m1.checkbox("AVS match", key="in_avs")
        cvv = m2.checkbox("CVV passed", key="in_cvv")
        three_ds = m3.checkbox("3-D Secure", key="in_tds")
        promo = m4.checkbox("Promo used", key="in_promo")

        submitted = st.form_submit_button("Score transaction", type="primary")

row = {
    "account_age_days": age, "total_transactions_user": n_tx, "avg_amount_user": avg_amount,
    "amount": amount, "country": country, "bin_country": bin_country, "channel": channel,
    "merchant_category": category, "promo_used": int(promo), "avs_match": int(avs),
    "cvv_result": int(cvv), "three_ds_flag": int(three_ds), "shipping_distance_km": distance,
    "transaction_time": f"{date.isoformat()}T{hour:02d}:00:00Z",
}

with right:
    if not submitted:
        st.markdown(
            '<div class="card"><h4>Result</h4>Fill in the transaction and press '
            "<b>Score transaction</b>, or load an example from the sidebar.</div>",
            unsafe_allow_html=True,
        )
    else:
        res = score_transaction(pipeline, meta, row, threshold)
        p = res["probability"]
        high = res["label"] == "High Risk"
        pos = min(max(p, 0.0), 1.0) * 100
        thr_pos = threshold * 100
        st.markdown(
            f"""
            <div class="result {'high' if high else 'low'}">
                <div class="label">Estimated fraud probability</div>
                <div class="value">{p:.1%}</div>
                <div class="tag">{'⚠️ High Risk' if high else '✅ Low Risk'}</div>
                <div class="meter">
                    <div class="thr" style="left:{thr_pos:.1f}%"></div>
                    <div class="pin" style="left:calc({pos:.1f}% - 2px)"></div>
            </div>
            <div class="scale"><span>0%</span><span>threshold {threshold:.0%}</span><span>100%</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        contrib = res["contributions"]
        contrib = contrib[contrib["effect"].abs() >= 0.001].head(7)
        st.markdown("##### What drives this score")
        if contrib.empty:
            st.write("No single input moves the score noticeably compared with a typical transaction.")
        else:
            peak = float(contrib["effect"].abs().max())
            html = ""
            for _, r in contrib.iterrows():
                w = abs(r["effect"]) / peak * 50
                color = "#dc2f02" if r["effect"] > 0 else "#2fb36d"
                left_pct = 50 if r["effect"] > 0 else 50 - w
                html += (
                    f'<div class="row"><div class="name">{r["field"]}</div>'
                    f'<div class="track"><div class="bar" style="left:{left_pct:.1f}%;width:{w:.1f}%;'
                    f'background:{color}"></div></div>'
                    f'<div class="num">{r["effect"] * 100:+.1f} pts</div></div>'
                )
            st.markdown(
                f'<div class="card">{html}<div class="legend">Red raises the risk, green lowers it, compared '
                "with a typical legitimate transaction. Inputs are correlated, so effects can offset each other."
                "</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="note">This score is an estimate, not proof of fraud. '
            "High Risk transactions need human review before any action is taken.</div>",
            unsafe_allow_html=True,
        )