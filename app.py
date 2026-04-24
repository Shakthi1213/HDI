import streamlit as st
import numpy as np
import pickle

# ──────────────────────────────────────────────
# Load trained model and scaler
# ──────────────────────────────────────────────
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ──────────────────────────────────────────────
# App config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="GNI per Capita Predictor",
    page_icon="🌍",
    layout="centered",
)

st.title("🌍 GNI per Capita Predictor")

st.markdown("""
Predict a country's *GNI per Capita (USD)* based on:
- Financial infrastructure  
- Digital connectivity  
""")

st.divider()

# ───────────────── INPUT SECTION ─────────────────
st.subheader("📥 Enter Country Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    atms = st.number_input(
        "ATMs per 100k people",
        min_value=0.0, max_value=500.0,
        value=30.0, step=1.0
    )

with col2:
    internet = st.number_input(
        "Internet Users (%)",
        min_value=0.0, max_value=100.0,
        value=50.0, step=0.5
    )

with col3:
    mobile = st.number_input(
        "Mobile Subscriptions per 100",
        min_value=0.0, max_value=300.0,
        value=80.0, step=1.0
    )

# Debug (optional - remove later)
st.write("Inputs:", atms, internet, mobile)

st.divider()

# ───────────────── PREDICTION ─────────────────
if st.button("🔮 Predict GNI per Capita", use_container_width=True):

    input_data = np.array([[atms, internet, mobile]])

    # Apply scaling
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prediction = max(0, prediction)

    st.success(f"### 💰 Predicted GNI per Capita: *${prediction:,.0f} USD*")

    # ───────── Interpretation ─────────
    st.subheader("📊 Interpretation")

    if prediction < 5000:
        st.markdown("🔴 *Low Income Economy*")
    elif prediction < 15000:
        st.markdown("🟡 *Lower-Middle Income Economy*")
    elif prediction < 40000:
        st.markdown("🟢 *Upper-Middle / High Income Economy*")
    else:
        st.markdown("🔵 *Very High Income Economy*")

st.divider()
