import streamlit as st
import requests

st.set_page_config(page_title="Customer Churn Prediction")

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict churn risk.")

# --- Inputs ---
tenure = st.number_input("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

# --- Predict button ---
if st.button("Predict Churn"):

    payload = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    try:
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()

            st.success(f"Prediction: {result['prediction']}")
            st.info(f"Churn Probability: {result['churn_probability']:.2f}")

        else:
            st.error(f"API Error: {response.text}")

    except Exception as e:
        st.error(f"Connection error: {e}")