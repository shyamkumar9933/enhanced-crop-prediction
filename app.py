from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from predict import load_model, top_k_recommendations

st.set_page_config(page_title="Crop Prediction", page_icon="🌾", layout="centered")

st.title("🌾 Enhanced Crop Prediction System")
st.caption("Top-3 crop recommendations from soil and environmental conditions")

try:
    model = load_model(ROOT / "models" / "rf_optuna.joblib")
except FileNotFoundError:
    st.error("Model file not found. Run: python src/train.py")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    N = st.number_input("Nitrogen (N)", 0.0, 140.0, 50.0)
    P = st.number_input("Phosphorus (P)", 5.0, 145.0, 53.0)
    K = st.number_input("Potassium (K)", 5.0, 205.0, 48.0)
    temperature = st.number_input("Temperature (°C)", 8.82, 43.67, 25.0)
with col2:
    humidity = st.number_input("Humidity (%)", 14.25, 99.98, 71.0)
    ph = st.number_input("Soil pH", 3.5, 9.93, 6.46)
    rainfall = st.number_input("Rainfall (mm)", 20.21, 298.56, 103.0)

if st.button("Predict Top 3 Crops", type="primary"):
    values = {
        "N": N, "P": P, "K": K, "temperature": temperature,
        "humidity": humidity, "ph": ph, "rainfall": rainfall
    }
    results = top_k_recommendations(model, values, k=3)

    st.subheader("Recommendations")
    for rank, item in enumerate(results, 1):
        st.write(f"**{rank}. {item['crop']}** — {item['confidence']:.2f}% confidence")

    st.info("The confidence values are model probabilities, not guarantees of yield.")
