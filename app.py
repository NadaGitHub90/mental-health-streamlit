import streamlit as st
import pandas as pd
import joblib, json

# 1. Load raw survey data
df = pd.read_csv("mmc2.csv")

# 2. Load model artifacts
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
with open("top7_features.json") as f:
    features = json.load(f)

# 3. Your scoring map
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
st.write("Answer the questions below, then click **Predict**:")

# 4. Collect user answers into a dict
user_dict = {}
for question in features:
    options = df[question].dropna().unique().tolist()
    options.sort()
    choice = st.selectbox(question, options)
    user_dict[question] = score_map.get(choice, 0.0)

# 5. Wait for button click
if st.button("Predict"):
    # only run this block once “Predict” is pressed
    input_df = pd.DataFrame([user_dict])
    scaled   = scaler.transform(input_df)
    pred     = model.predict(scaled)[0]

    st.subheader("Prediction Result:")
    if pred == "at risk":
        st.error("🚨 This person is likely AT RISK 🚨")
    else:
        st.success("✔️ This person is likely LOW RISK ✔️")



