import streamlit as st
import pandas as pd
import joblib, json

# 1. Load the raw survey data (make sure you add mmc2.csv to your repo root!)
df = pd.read_csv("mmc2.csv")

# 2. Load model artifacts
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
with open("top7_features.json") as f:
    features = json.load(f)

# 3. Define the same mapping from strings → risk scores
score_map = {
    "Not at all":            0.01,
    "Never":                 0.01,
    "Rarely":                0.25,
    "Sometimes":             0.4,
    "Several days":          0.5,
    "Half of days":          0.7,
    "Often":                 0.8,
    "Nearly everyday":       0.9,
    "Most of the times":     0.85,
    "All the times":         1.0,
    "Always":                1.0,
    "Less then once a week": 0.2,
    "Not during last month": 0.01,
    "Three or more in week": 0.5,
    "Once or twice a week":  0.3
}

st.title("Mental Health Risk Prediction")
st.write("Please answer the following questions:")

# 4. Build a dict of numeric scores by asking each question
user_dict = {}
for question in features:
    # fetch the actual string options from your dataset
    options = df[question].dropna().unique().tolist()
    options.sort()  # alphabetical, or remove if you want original order
    
    # show the select box
    choice = st.selectbox(question, options)
    
    # map that string to its numeric score
    user_dict[question] = score_map.get(choice, 0.0)

# 5. Predict
input_df = pd.DataFrame([user_dict])
scaled   = scaler.transform(input_df)
pred     = model.predict(scaled)[0]

st.subheader("Prediction Result:")
if pred == "at risk":
    st.error("🚨 This person is likely AT RISK 🚨")
else:
    st.success("✔️ This person is likely AT LOW RISK ✔️")


