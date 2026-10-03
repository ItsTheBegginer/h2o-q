"""Water-quality estimator: temperature, TDS, turbidity -> BOD and COD (estimate + 80% range).

Run:  streamlit run app.py
Needs in the same folder: bod_lo/mid/hi.json, cod_lo/mid/hi.json, model_meta.json
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

HERE = Path(__file__).parent
FEATS = ["temperature", "tds", "turbidity"]
LOG_FILE = HERE / "paired_readings.csv"

st.set_page_config(page_title="AquaEstimate | BOD & COD", page_icon="💧", layout="centered")

CSS = """
<style>
/* ── Base ── */
.stApp { background: #f8fafc; }
.block-container { max-width: 820px; padding-top: 1.5rem; }

/* ── Header ── */
.app-header { padding: 1rem 0; border-bottom: 2px solid #2563eb; margin-bottom: 1.5rem; }
.app-header h2 { font-size: 1.4rem; font-weight: 700; color: #0f172a; margin: 0; }
.app-header p  { font-size: 0.85rem; color: #64748b; margin: 2px 0 0 0; }

/* ── Section label ── */
.section-label {
    font-size: .7rem; font-weight: 700; letter-spacing: .1em;
    color: #94a3b8; text-transform: uppercase; margin-bottom: 8px;
}

/* ── Result cards ── */
.result-card {
    background: #ffffff; border-radius: 16px; padding: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08); border: 1px solid #e2e8f0;
}
.result-card .param-tag {
    display: inline-block; font-size: .72rem; font-weight: 700;
    letter-spacing: .08em; color: #2563eb; background: #eff6ff;
    border-radius: 99px; padding: 2px 9px; margin-bottom: 6px;
}
.result-card .param-name { font-size: .95rem; color: #475569; margin-bottom: 6px; }
.result-card .value { font-size: 2.8rem; font-weight: 800; color: #0f172a; line-height: 1.1; }
.result-card .unit  { font-size: .9rem; color: #64748b; margin-left: 5px; }

/* ── Range bar ── */
.range-bar-track {
    height: 6px; background: #e2e8f0; border-radius: 99px;
    position: relative; margin: 12px 0 4px 0;
}
.range-bar-fill {
    background: #2563eb; height: 100%; border-radius: 99px;
}
.range-dot {
    width: 14px; height: 14px; background: #2563eb; border-radius: 50%;
    position: absolute; top: -4px; transform: translateX(-50%);
}
.range-ends {
    display: flex; justify-content: space-between;
    font-size: .78rem; color: #94a3b8;
}

/* ── Inline warning ── */
.inline-warn {
    background: #fffbeb; border-left: 3px solid #f59e0b;
    padding: 6px 10px; border-radius: 6px; font-size: .85rem;
    color: #92400e; margin: 4px 0;
}

/* ── Number input labels ── */
div[data-testid="stNumberInput"] label p { font-weight: 600; color: #1e293b; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    meta = json.load(open(HERE / "model_meta.json"))
    models = {}
    for target in ("bod", "cod"):
        for q in ("lo", "mid", "hi"):
            m = xgb.XGBRegressor()
            m.load_model(str(HERE / f"{target}_{q}.json"))
            models[(target, q)] = m
    return meta, models


def predict(models, target, row):
    raw = [float(np.expm1(models[(target, q)].predict(row))[0]) for q in ("lo", "mid", "hi")]
    lo, mid, hi = np.clip(np.sort(raw), 0, None)  # sort prevents crossing quantiles
    return lo, mid, hi


def classify(val, target):
    if target == "bod":
        if val < 2:   return "Excellent", "#16a34a"
        if val < 6:   return "Good",      "#0d9488"
        if val < 30:  return "Moderate",  "#d97706"
        return "Poor", "#dc2626"
    else:  # cod
        if val < 40:  return "Clean",     "#16a34a"
        if val < 100: return "Moderate",  "#d97706"
        if val < 200: return "High",      "#ea580c"
        return "Very High", "#dc2626"


def result_card(title, tag, lo, mid, hi, target):
    pos = 50.0 if hi - lo < 1e-9 else float(np.clip((mid - lo) / (hi - lo) * 100, 3, 97))
    label, badge_color = classify(mid, target)
    badge = (
        f'<span style="background:{badge_color}20; color:{badge_color}; '
        f'border:1px solid {badge_color}40; border-radius:99px; '
        f'padding:2px 10px; font-size:.8rem; font-weight:600;">{label}</span>'
    )
    return f"""
    <div class="result-card">
      <div class="param-tag">{tag}</div>
      <div class="param-name">{title}</div>
      <div>
        <span class="value">{mid:.1f}</span><span class="unit">mg/L</span>
      </div>
      <div class="range-bar-track">
        <div class="range-dot" style="left:{pos}%"></div>
      </div>
      <div class="range-ends">
        <span>{lo:.1f}</span>
        <span>{hi:.1f}</span>
      </div>
      <div style="margin-top:10px;">{badge}</div>
    </div>"""


try:
    meta, models = load_models()
except Exception as e:
    st.error(f"Could not load model files from {HERE}. Details: {e}")
    st.stop()

lo_bound = np.array(meta["bod"]["train_min"])
hi_bound = np.array(meta["bod"]["train_max"])

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """<div class="app-header">
      <h2>💧 AquaEstimate</h2>
      <p>Estimate BOD &amp; COD from three sensor readings</p>
    </div>""",
    unsafe_allow_html=True,
)

# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Sensor readings</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
vals = {
    "temperature": c1.number_input("Temperature (°C)", min_value=0.0, value=22.0, step=0.1, format="%.2f"),
    "tds":         c2.number_input("TDS (ppm)",        min_value=0.0, value=200.0, step=1.0, format="%.2f"),
    "turbidity":   c3.number_input("Turbidity (NTU)",  min_value=0.0, value=5.0,   step=0.1, format="%.2f"),
}

# Range hints below each column
hint_labels = ["temperature", "tds", "turbidity"]
hint_cols   = [c1, c2, c3]
for col, feat, lo_b, hi_b in zip(hint_cols, hint_labels, lo_bound, hi_bound):
    col.markdown(
        f'<p style="font-size:.75rem; color:#94a3b8; margin-top:-8px;">'
        f'Range: {lo_b:.1f} – {hi_b:.1f}</p>',
        unsafe_allow_html=True,
    )

row = pd.DataFrame([[vals[f] for f in FEATS]], columns=FEATS)

# ── Out-of-range + zero-value warnings ────────────────────────────────────────
names = {"temperature": "Temperature", "tds": "TDS", "turbidity": "Turbidity"}
for i, f in enumerate(FEATS):
    if vals[f] < lo_bound[i] or vals[f] > hi_bound[i]:
        st.markdown(
            f'<div class="inline-warn">'
            f'<b>{names[f]}</b> = {vals[f]} is outside the training range '
            f'({lo_bound[i]:.1f} – {hi_bound[i]:.1f}). The estimate may be unreliable.'
            f'</div>',
            unsafe_allow_html=True,
        )

if vals["tds"] == 0:
    st.markdown(
        '<div class="inline-warn">TDS of 0 may indicate an invalid reading.</div>',
        unsafe_allow_html=True,
    )
if vals["turbidity"] == 0:
    st.markdown(
        '<div class="inline-warn">Turbidity of 0 may indicate an invalid reading.</div>',
        unsafe_allow_html=True,
    )

# ── Timestamp ─────────────────────────────────────────────────────────────────
st.markdown(
    f'<p style="font-size:.78rem; color:#94a3b8; margin: 8px 0 16px 0;">'
    f'Last updated: {datetime.now().strftime("%H:%M:%S")}</p>',
    unsafe_allow_html=True,
)

# ── Results ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Estimated results</div>', unsafe_allow_html=True)
results = {t: predict(models, t, row) for t in ("bod", "cod")}

left, right = st.columns(2)
left.markdown(
    result_card("Biochemical Oxygen Demand", "BOD", *results["bod"], "bod"),
    unsafe_allow_html=True,
)
right.markdown(
    result_card("Chemical Oxygen Demand", "COD", *results["cod"], "cod"),
    unsafe_allow_html=True,
)

# ── Log readings ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:1.8rem;">Log a reading</div>', unsafe_allow_html=True)
with st.expander("Save this reading with lab results (for future retraining)"):
    a, b = st.columns(2)
    lab_bod = a.number_input("Lab BOD (mg/L)", min_value=0.0, value=0.0, step=0.1)
    lab_cod = b.number_input("Lab COD (mg/L)", min_value=0.0, value=0.0, step=0.1)
    note = st.text_input("Site / note")
    if st.button("Save reading", type="primary"):
        rec = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **{f: vals[f] for f in FEATS},
            "lab_bod": lab_bod or None,
            "lab_cod": lab_cod or None,
            "pred_bod": round(results["bod"][1], 2),
            "pred_cod": round(results["cod"][1], 2),
            "note": note,
        }
        pd.DataFrame([rec]).to_csv(LOG_FILE, mode="a", header=not LOG_FILE.exists(), index=False)
        st.success(f"Saved to {LOG_FILE.name}")

    if LOG_FILE.exists():
        df_log = pd.read_csv(LOG_FILE)
        st.markdown('<div class="section-label">Last 5 readings</div>', unsafe_allow_html=True)
        st.dataframe(df_log.tail(5), use_container_width=True)

# Download button outside the expander, always visible when log exists
if LOG_FILE.exists():
    st.download_button(
        "⬇️ Download logged readings",
        LOG_FILE.read_bytes(),
        "paired_readings.csv",
    )
    st.caption("On hosted platforms this file may be wiped on restart. Download it regularly.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """<div style="text-align:center; color:#94a3b8; font-size:.8rem;
                  margin-top:2rem; padding-top:1rem; border-top:1px solid #e2e8f0;">
    Predictions are estimates based on ML models. Not a substitute for lab analysis.
    </div>""",
    unsafe_allow_html=True,
)
