import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/loan_model.pkl")

st.title("CreditWise Loan Approval System")

income = st.number_input("Applicant Income")
loan_amount = st.number_input("Loan Amount")

if st.button("Predict"):
    data = pd.DataFrame([[income, loan_amount]], 
                        columns=["ApplicantIncome", "LoanAmount"])

    prediction = model.predict(data)

    st.write("Loan Status:", prediction[0])
