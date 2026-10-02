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
.stApp { background: linear-gradient(160deg, #eaf6ff 0%, #f4efff 55%, #fff0f6 100%); }
.block-container { padding-top: 1.5rem; max-width: 820px; }
.hero { background: linear-gradient(120deg, #0ea5e9 0%, #6366f1 55%, #d946ef 100%);
        border-radius: 22px; padding: 28px 30px; color: #fff; margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(99,102,241,.30); }
.hero h1 { margin: 0; font-size: 2.1rem; color: #fff; }
.hero p { margin: 6px 0 0 0; opacity: .92; font-size: 1.02rem; }
.section { font-weight: 700; font-size: 1.15rem; color: #1e293b; margin: 18px 0 8px 0; }
.card { background: #fff; border-radius: 20px; padding: 22px 22px 18px 22px;
        box-shadow: 0 6px 20px rgba(15,23,42,.08); border-top: 6px solid var(--c1); }
.card .tag { font-size: .8rem; font-weight: 700; letter-spacing: .08em; color: var(--c1); }
.card .name { font-size: 1.05rem; color: #475569; margin-bottom: 4px; }
.card .big { font-size: 3rem; font-weight: 800; line-height: 1.1; color: #0f172a; }
.card .unit { font-size: 1rem; font-weight: 600; color: #64748b; margin-left: 6px; }
.card .rng { margin: 10px 0 12px 0; color: #334155; font-size: .98rem; }
.bar { position: relative; height: 12px; border-radius: 99px;
       background: linear-gradient(90deg, var(--c1), var(--c2)); opacity: .9; }
.dot { position: absolute; top: -5px; width: 22px; height: 22px; border-radius: 50%;
       background: #fff; border: 4px solid #0f172a; transform: translateX(-50%); }
.ends { display: flex; justify-content: space-between; font-size: .8rem; color: #64748b; margin-top: 8px; }
.note { background: rgba(255,255,255,.75); border-left: 5px solid #6366f1; border-radius: 12px;
        padding: 12px 16px; color: #334155; font-size: .93rem; margin-top: 18px; }
.chip { display:inline-block; background:#fff; color:#4338ca; border-radius:99px; padding:3px 12px;
        font-size:.8rem; font-weight:600; margin-right:6px; box-shadow:0 2px 8px rgba(0,0,0,.06); }
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


def result_card(title, tag, lo, mid, hi, c1, c2):
    pos = 50.0 if hi - lo < 1e-9 else float(np.clip((mid - lo) / (hi - lo) * 100, 3, 97))
    return f"""
    <div class="card" style="--c1:{c1};--c2:{c2}">
      <div class="tag">{tag}</div>
      <div class="name">{title}</div>
      <div><span class="big">{mid:.1f}</span><span class="unit">mg/L (estimate)</span></div>
      <div class="rng">80% range: <b>{lo:.1f} to {hi:.1f}</b> mg/L</div>
      <div class="bar"><div class="dot" style="left:{pos}%"></div></div>
      <div class="ends"><span>{lo:.1f}</span><span>estimate</span><span>{hi:.1f}</span></div>
    </div>"""


try:
    meta, models = load_models()
except Exception as e:
    st.error(f"Could not load model files from {HERE}. Details: {e}")
    st.stop()

lo_bound = np.array(meta["bod"]["train_min"])
hi_bound = np.array(meta["bod"]["train_max"])

# ---- Header ---------------------------------------------------------------------
st.markdown(
    """<div class="hero"><h1>💧 AquaEstimate</h1>
    <p>Estimate BOD and COD from three quick sensor readings.</p></div>""",
    unsafe_allow_html=True,
)

# ---- Inputs ---------------------------------------------------------------------
st.markdown('<div class="section">🧪 Sensor readings</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
vals = {
    "temperature": c1.number_input("🌡️ Temperature (°C)", min_value=0.0, value=22.0, step=0.1, format="%.2f"),
    "tds": c2.number_input("🧂 TDS (ppm)", min_value=0.0, value=200.0, step=1.0, format="%.2f"),
    "turbidity": c3.number_input("🌫️ Turbidity (NTU)", min_value=0.0, value=5.0, step=0.1, format="%.2f"),
}
row = pd.DataFrame([[vals[f] for f in FEATS]], columns=FEATS)

names = {"temperature": "Temperature", "tds": "TDS", "turbidity": "Turbidity"}
for i, f in enumerate(FEATS):
    if vals[f] < lo_bound[i] or vals[f] > hi_bound[i]:
        st.warning(
            f"{names[f]} = {vals[f]} is outside the training range "
            f"({lo_bound[i]:.1f} to {hi_bound[i]:.1f}). The estimate may be unreliable."
        )

# ---- Results --------------------------------------------------------------------
st.markdown('<div class="section">📊 Estimated results</div>', unsafe_allow_html=True)
results = {t: predict(models, t, row) for t in ("bod", "cod")}
left, right = st.columns(2)
left.markdown(
    result_card("Biochemical Oxygen Demand", "BOD", *results["bod"], "#0ea5e9", "#6366f1"),
    unsafe_allow_html=True,
)
right.markdown(
    result_card("Chemical Oxygen Demand", "COD", *results["cod"], "#d946ef", "#f43f5e"),
    unsafe_allow_html=True,
)

st.markdown(
    """<div class="note"><b>How to read this.</b> The marker is the best estimate and the bar is
    the range the model expects the true value to fall in about 80% of the time. The model uses
    only temperature, TDS and turbidity, so ranges are wide. Treat results as rough indications,
    not laboratory measurements.</div>""",
    unsafe_allow_html=True,
)

# ---- Optional: log readings with lab results ---------------------------------------
st.markdown('<div class="section">📝 Log a reading</div>', unsafe_allow_html=True)
with st.expander("Save this reading with lab results (for future retraining)"):
    a, b = st.columns(2)
    lab_bod = a.number_input("Lab BOD (mg/L)", min_value=0.0, value=0.0, step=0.1)
    lab_cod = b.number_input("Lab COD (mg/L)", min_value=0.0, value=0.0, step=0.1)
    note = st.text_input("Site / note")
    if st.button("💾 Save reading", type="primary"):
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
        st.download_button("⬇️ Download logged readings", LOG_FILE.read_bytes(), "paired_readings.csv")
        st.caption("On hosted platforms this file may be wiped on restart. Download it regularly.")