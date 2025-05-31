import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("tb_predictor_model.pkl")

# Define the input features in the same order used during training
feature_names = [
    "fever for two weeks",
    "coughing blood",
    "sputum mixed with blood",
    "night sweats",
    "chest pain",
    "back pain in certain parts",
    "shortness of breath",
    "weight loss",
    "body feels tired",
    "lumps that appear around the armpits and neck",
    "cough and phlegm continuously for two weeks to four weeks",
    "swollen lymph nodes",
    "loss of appetite"
]

st.set_page_config(page_title="TB Prediction App", layout="centered")
st.title("🩺 Tuberculosis Prediction App")

st.markdown("""
This tool predicts the **likelihood of having Tuberculosis (TB)** based on symptoms.
Please answer the questions below as accurately as possible.
""")

# Collect input for all symptoms
user_input = []
for symptom in feature_names:
    val = st.selectbox(f"Do you have: {symptom}?", ["No", "Yes"])
    user_input.append(1 if val == "Yes" else 0)

# Predict and display result
if st.button("Predict TB Status"):
    prediction = model.predict([user_input])[0]
    proba = model.predict_proba([user_input])[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ The model predicts that this person is **likely to have TB** (confidence: {proba:.2%}). Please consult a doctor.")
    else:
        st.success(f"✅ The model predicts that this person is **unlikely to have TB** (confidence: {proba:.2%}).")
