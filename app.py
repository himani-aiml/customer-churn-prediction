"""
Customer Churn Prediction — Streamlit App
Loads the trained pipeline (churnshield_model_pipeline.pkl) saved by the notebook
and predicts churn probability for a single customer entered via a form.

Run locally with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="ChurnShield — Customer Retention Predictor", page_icon="📉", layout="centered")

st.title("📉 ChurnShield — Customer Retention Predictor")
st.write(
    "Enter a customer's details to predict the probability that they will churn. "
    "This uses a Random Forest model trained on the IBM Telco Customer Churn dataset."
)


@st.cache_resource
def load_model():
    return joblib.load("churnshield_model_pipeline.pkl")


model = load_model()

st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

with col2:
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)

# TotalCharges is usually close to tenure * monthly_charges for an existing customer
total_charges = st.number_input(
    "Total Charges ($) so far", min_value=0.0, value=round(tenure * monthly_charges, 2)
)

if st.button("Predict Churn", type="primary"):
    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }])

    prob = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk — {prob*100:.1f}% probability of churn")
        st.write("**Recommendation:** Prioritize this customer for a retention offer "
                 "(e.g., contract upgrade discount, free tech support trial).")
    else:
        st.success(f"✅ Low Risk — {prob*100:.1f}% probability of churn")
        st.write("**Recommendation:** No immediate action needed; continue standard engagement.")

    st.progress(min(int(prob * 100), 100))

st.divider()
st.caption(
    "Model: Random Forest (tuned with GridSearchCV, trained with SMOTE for class imbalance) · "
    "Dataset: IBM Telco Customer Churn"
)
