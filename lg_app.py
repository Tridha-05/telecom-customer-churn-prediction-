import streamlit as st
import pandas as pd
import pickle


bundle = pickle.load(open("churn_model_bundle.pkl", "rb"))

model = bundle["model"]
scaler = bundle["scaler"]
columns = bundle["columns"]
threshold = bundle["threshold"]


st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("Telecom Customer Churn Predictor")
st.write("This tool estimates the risk of a customer leaving their mobile or internet service provider.")

st.subheader("Customer Details")
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
Partner = st.selectbox("Has a Partner", ["No", "Yes"])
Dependents = st.selectbox("Has Dependents", ["No", "Yes"])
tenure = st.number_input("How long have you been with the company(months)",min_value=0,step=1)

st.subheader("Services Used")
PhoneService = st.selectbox("Uses Phone Service",["No","Yes"])
MultipleLines = st.selectbox("Multiple Phone Numbers",["No", "Yes", "No phone service"])
InternetService = st.selectbox("Internet Service Type",["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Extra Internet Protection",["No", "Yes"],help="Example: antivirus, firewall, or security add-ons from your provider")
TechSupport = st.selectbox("Technical Support Service",["No", "Yes"])

st.subheader("Billing Information")
Contract = st.selectbox("Contract Type",["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Receive Bills Online",["No", "Yes"])
PaymentMethod = st.selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
MonthlyCharges = st.number_input("Monthly Charges ($)",min_value=0.0)
TotalCharges = st.number_input("Total Charges Paid So Far ($)",min_value=0.0)

if st.button("Predict Customer Churn Risk"):
    st.write("Processing prediction...")
    SeniorCitizen = 1 if SeniorCitizen == "Yes" else 0
    user_data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'TechSupport': TechSupport,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    input_df = pd.DataFrame([user_data])

    # one hot enocding
    input_df = pd.get_dummies(input_df)

    #match index with dataset
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # scale numeric values
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    prob = model.predict_proba(input_df)[0][1]
    display_prob=min(prob*100,99.9)

    if prob>=threshold:
        st.error(f"High Churn Risk ({display_prob:.1f}%)")
    else:
        st.success(f"Low Churn Risk ({display_prob:.1f}%)")

    st.caption("This prediction is based on historical customer patterns, not personal intent.")