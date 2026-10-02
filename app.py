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
LABELS = {
    "temperature": "Temperature (°C)",
    "tds": "TDS (ppm / mg/L)",
    "turbidity": "Turbidity (NTU)",
}
LOG_FILE = HERE / "paired_readings.csv"

st.set_page_config(page_title="BOD & COD Estimator", page_icon="💧", layout="centered")


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


def category(value, edges):
    if value < edges[0]:
        return "Low", "🟢"
    if value < edges[1]:
        return "Medium", "🟠"
    return "High", "🔴"


try:
    meta, models = load_models()
except Exception as e:
    st.error(f"Could not load model files from {HERE}. Details: {e}")
    st.stop()

st.title("💧 BOD & COD Estimator")
st.caption(
    "Rough estimates from three sensor readings. Not a substitute for laboratory "
    "analysis. Each result comes with an 80% range."
)

# ---- Inputs -----------------------------------------------------------------
lo_bound = np.array(meta["bod"]["train_min"])
hi_bound = np.array(meta["bod"]["train_max"])
c1, c2, c3 = st.columns(3)
vals = {}
for col, name, i in zip((c1, c2, c3), FEATS, range(3)):
    vals[name] = col.number_input(LABELS[name], value=float(np.round(np.median([lo_bound[i], hi_bound[i]]), 1)),
                                  min_value=0.0, step=0.1, format="%.2f")

row = pd.DataFrame([[vals[f] for f in FEATS]], columns=FEATS)

# Out-of-range warnings
for i, f in enumerate(FEATS):
    if vals[f] < lo_bound[i] or vals[f] > hi_bound[i]:
        st.warning(
            f"{LABELS[f]} = {vals[f]} is outside the training range "
            f"({lo_bound[i]:.1f} to {hi_bound[i]:.1f}). The estimate may be unreliable."
        )

# ---- Sidebar: category thresholds ----------------------------------------------
st.sidebar.header("Category thresholds")
st.sidebar.caption(
    "Defaults are tertiles of the training data (placeholders). "
    "Replace them with limits from the water-quality standard you follow."
)
thresholds = {}
for target in ("bod", "cod"):
    e = meta[target]["edges"]
    a = st.sidebar.number_input(f"{target.upper()}: Low below", value=float(round(e[0], 1)), step=0.5, key=f"{target}_a")
    b = st.sidebar.number_input(f"{target.upper()}: High from", value=float(round(e[1], 1)), step=0.5, key=f"{target}_b")
    thresholds[target] = (a, b)

# ---- Results --------------------------------------------------------------------
st.subheader("Estimates")
results = {}
cols = st.columns(2)
for col, target in zip(cols, ("bod", "cod")):
    lo, mid, hi = predict(models, target, row)
    results[target] = (lo, mid, hi)
    label, icon = category(mid, thresholds[target])
    with col:
        st.metric(f"{target.upper()} (mg/L, estimated)", f"{mid:.1f}")
        st.write(f"80% range: **{lo:.1f} – {hi:.1f}**")
        st.write(f"Category: {icon} **{label}**")

st.info(
    "How to read this: the range is the useful part. With only temperature, TDS and "
    "turbidity the model reduces typical error by roughly 13–16% compared with guessing "
    "the median, so the ranges are wide."
)

# ---- Optional: log ESP32 + lab pairs for future retraining -------------------------
with st.expander("Log a reading with lab results (for future retraining)"):
    lab_bod = st.number_input("Lab BOD (mg/L)", min_value=0.0, value=0.0, step=0.1)
    lab_cod = st.number_input("Lab COD (mg/L)", min_value=0.0, value=0.0, step=0.1)
    note = st.text_input("Site / note")
    if st.button("Save reading"):
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
        st.download_button("Download logged readings", LOG_FILE.read_bytes(), "paired_readings.csv")
        st.caption("On hosted platforms this file may be wiped on restart. Download it regularly.")
