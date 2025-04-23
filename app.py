import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

# Load saved files
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
with open("top7_features.json", "r") as f:
    features = json.load(f)

st.title("Mental Health Risk Prediction")
st.write("This tool uses survey responses to predict if someone is at high or low risk for mental health challenges based on social media use.")

user_inputs = {}

# Dynamically create input fields for the 7 features
for feat in features:
    if "age" in feat.lower():
        user_inputs[feat] = st.slider("Your Age", 10, 80, 25)
    elif "income" in feat.lower():
        user_inputs[feat] = st.selectbox("Monthly Income (0=No Income to 4=High)", [0, 1, 2, 3, 4])
    elif "education" in feat.lower():
        user_inputs[feat] = st.selectbox("Education (0=No School to 5=Graduate)", [0, 1, 2, 3, 4, 5])
    else:
        user_inputs[feat] = st.selectbox(f"{feat}", [0, 1, 2, 3])

# Prediction
if st.button("Predict"):
    input_df = pd.DataFrame([user_inputs])
    scaled = scaler.transform(input_df)
    pred = model.predict(scaled)[0]
    st.subheader("Prediction:")
    st.success(f"The person is likely at **{pred.upper()}** mental health risk.")

