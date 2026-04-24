import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ──────────────────────────────────────────────
# Pre-trained model coefficients (from training on merged_data_with_all_factors.xlsx)
# No Excel file needed at runtime.
# ──────────────────────────────────────────────

# --- Linear Regression (hardcoded weights) ---
lr_model = LinearRegression()
lr_model.coef_ = np.array([-167.61339508, 712.9267099, 76.60924982])
lr_model.intercept_ = -44814.47043637364

# Feature names (in order)
FEATURES = ["ATMs_per100k", "Internet_Users_pct", "Mobile_Subs_per100"]

# ──────────────────────────────────────────────
# App layout
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="GNI per Capita Predictor",
    page_icon="🌍",
    layout="centered",
)

st.title("🌍 GNI per Capita Predictor")
st.markdown(
    """
    Predict a country's **GNI per Capita (USD)** based on key financial &
    digital infrastructure indicators using a **Linear Regression** model
    trained on World Bank data.
    """
)

st.divider()

# ── Input Section ──────────────────────────────
st.subheader("📥 Enter Country Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    atms = st.number_input(
        "ATMs per 100k people",
        min_value=0.0,
        max_value=500.0,
        value=30.0,
        step=1.0,
        help="Number of ATMs per 100,000 adults",
    )

with col2:
    internet = st.number_input(
        "Internet Users (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=0.5,
        help="Percentage of population using the internet",
    )

with col3:
    mobile = st.number_input(
        "Mobile Subscriptions per 100",
        min_value=0.0,
        max_value=300.0,
        value=80.0,
        step=1.0,
        help="Mobile cellular subscriptions per 100 people",
    )
st.write("Inputs:", atms, internet, mobile)
st.divider()

# ── Prediction ─────────────────────────────────
if st.button("🔮 Predict GNI per Capita", use_container_width=True, type="primary"):
    input_data = np.array([[atms, internet, mobile]])
    prediction = lr_model.predict(input_data)[0]
    prediction = max(0, prediction)  # GNI cannot be negative

    st.success(f"### Predicted GNI per Capita: **${prediction:,.0f} USD**")

    # ── Interpretation ──
    st.subheader("📊 Interpretation")

    if prediction < 5000:
        level = "🔴 Low Income"
        desc = "Typical of low-income economies with limited financial access."
    elif prediction < 15000:
        level = "🟡 Lower-Middle Income"
        desc = "Emerging economies with growing digital infrastructure."
    elif prediction < 40000:
        level = "🟢 Upper-Middle / High Income"
        desc = "Well-developed economies with strong digital penetration."
    else:
        level = "🔵 Very High Income"
        desc = "Highly developed economies with advanced financial systems."

    st.markdown(f"**Income Level:** {level}")
    st.markdown(f"{desc}")

    # ── Feature Contribution ──
    st.subheader("🧮 Feature Contributions to Prediction")
    contributions = {
        "ATMs per 100k": lr_model.coef_[0] * atms,
        "Internet Users (%)": lr_model.coef_[1] * internet,
        "Mobile Subscriptions per 100": lr_model.coef_[2] * mobile,
        "Intercept (Baseline)": lr_model.intercept_,
    }

    for feat, val in contributions.items():
        sign = "+" if val >= 0 else ""
        st.markdown(f"- **{feat}**: {sign}{val:,.0f} USD")

st.divider()

# ── Model Info ────────────────────────────────
with st.expander("ℹ️ About this Model"):
    st.markdown(
        """
        **Model:** Linear Regression  
        **Target:** GNI per Capita (USD)  
        **Features used:**
        - `ATMs_per100k` — ATMs per 100,000 adults
        - `Internet_Users_pct` — % of population using the internet
        - `Mobile_Subs_per100` — Mobile subscriptions per 100 people

        **Training Data:** World Bank merged dataset (124 countries)  
        **Test R² Score:** 0.18  
        **Test RMSE:** ~28,217 USD

        > ⚠️ This model is trained on a small clean subset (10 rows after dropping NaNs).
        > Predictions are indicative and not meant for policy use.
        """
    )

with st.expander("📐 Model Coefficients"):
    st.markdown(
        f"""
        | Feature | Coefficient |
        |---|---|
        | ATMs per 100k | {lr_model.coef_[0]:.4f} |
        | Internet Users (%) | {lr_model.coef_[1]:.4f} |
        | Mobile Subs per 100 | {lr_model.coef_[2]:.4f} |
        | Intercept | {lr_model.intercept_:.4f} |
        """
    )
