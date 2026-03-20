import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Fraud Predictor", page_icon="🚨", layout="wide")

@st.cache_resource
def load_model():
    model_path = "/app/models/fraud_detection_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

st.title("💳Real-Time Fraud Predictor")
st.markdown("Enter transaction details below to get an instant prediction from the trained XGBoost model.")

if model is None:
    st.error("🚨 Model file not found at `/app/models/fraud_detection_model.pkl`. Please ensure the Airflow DAG has trained and saved the model successfully.")
    st.stop()

# Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Transaction Details")
    amt = st.number_input("Transaction Amount ($)", value=8500, min_value=0)
    amt_log = st.number_input("Amount Log (Auto-calculated if 0)", value=0)
    merchant = st.text_input("Merchant Name", value="fraud_store")
    category = st.selectbox("Category", ["shopping_net", "grocery_pos", "gas_transport", "misc_net", "misc_pos", "entertainment", "dining"])
    
with col2:
    st.subheader("Time & Location")
    trans_hour = st.slider("Transaction Hour (0-23)", 0, 23, 14)
    trans_day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
    trans_month = st.slider("Month", 1, 12, 12)
    distance_km = st.number_input("Distance from User (km)", value=50.0)
    state = st.text_input("State (Abbreviation)", value="NY")
    city_pop = st.number_input("City Population", value=500000)

with col3:
    st.subheader("User Demographics")
    age = st.number_input("Age", value=45, min_value=18, max_value=120)
    gender = st.selectbox("Gender", ["M", "F", "Other"])
    
    st.markdown("---")
    st.subheader("Engineered Flags")
    is_weekend = st.checkbox("Is Weekend?")
    is_night = st.checkbox("Is Nighttime?")
    category_risk = st.checkbox("High Category Risk?")
    distance_risk = st.checkbox("High Distance Risk?")

st.markdown("---")

if st.button("🔍Predict Fraud", type="primary", use_container_width=True):
    import math
    if amt_log == 0.0 and amt > 0:
        amt_log = math.log(amt)
        
    transaction = {
        'amt': [amt],
        'amt_log': [amt_log],
        'distance_km': [distance_km],
        'city_pop': [city_pop],
        'trans_hour': [trans_hour],
        'trans_day_of_week': [trans_day_of_week],
        'trans_month': [trans_month],
        'is_weekend': [1 if is_weekend else 0],
        'is_night': [1 if is_night else 0],
        'gender': [gender],
        'age': [age],
        'category': [category],
        'merchant': [merchant],
        'state': [state],
        'category_risk': [1 if category_risk else 0],
        'distance_risk': [1 if distance_risk else 0]
    }
    
    df = pd.DataFrame(transaction)
    try:
        probability = model.predict_proba(df)[0][1]
        threshold = 0.48
        is_fraud = probability >= threshold
        
        if is_fraud:
            st.error(f"### 🚨VERDICT: FRAUD DETECTED!\n**Probability Score:** {probability:.4f} (Threshold: {threshold})")
        else:
            st.success(f"### ✅VERDICT: LEGITIMATE\n**Probability Score:** {probability:.4f} (Threshold: {threshold})")
            
        with st.expander("View Raw Model Input"):
            st.json(transaction)
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")
