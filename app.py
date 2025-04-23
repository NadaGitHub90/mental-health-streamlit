import streamlit as st
import pandas as pd
import joblib, json

# 1) load your artifacts
model  = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# these are the *exact* feature names from top7_features.json
with open("top7_features.json") as f:
    features = json.load(f)

st.title("Mental Health Risk Prediction")
st.write("Answer the questions below, then click **Predict**")

# 2) manually re-implement the LabelEncoder mappings used in training
#    NOTE: these dicts map the *exact* strings in your data → the ints your RF saw
encodings = {
    "5. How long have you been using a social media account?": {
        "Less than 2-year": 0,
        "2-5 years":        1,
        "5-10 years":       2,
        "More than 10 years": 3
    },
    "7. How much time do you spend daily in social media?": {
        "Less than 1 hour": 0,
        "1-3 hours":        1,
        "3-5 hours":        2,
        "More than 5 hours":3
    },
    "10. How many friends do you know personally in social media? ": {
        "All of them": 0,
        "Few of them": 1,
        "Many of them":2,
        "Most of them":3
    },
    "16. Have you ever experienced peer pressure due to social media?": {
        "No":  0,
        "Yes": 1
    },
    "17. Does your emotion get influenced by other's posts (success, failure, loss)? ": {
        "Always":    0,
        "Not at all":1,
        "Sometimes":  2
    },
    "18. Have you ever compared yourself with other’s success or luxurious life?": {
        "All the times":     0,
        "Most of the times": 1,
        "Never":             2,
        "Sometimes":         3
    },
    "19. Do you think, your mental wellbeing would be better if you do not use social media?": {
        "No":  0,
        "Yes": 1
    },
    "25. Education": {
        "Graduate Level":   0,
        "Masters or above": 1,
        "Primary level":    2,
        "Secondary Level":  3
    },
    "27. Monthly income": {
        "10,000-40,000 Tk": 0,
        "40,000-70,000 Tk": 1,
        "Above 70,000 Tk":  2,
        "No income":        3
    }
    # (if your features list has any others, add them here the same way)
}

# 3) build the user‐input vector
user_vec = []
for feat in features:
    if feat.startswith("21. Please write your age"):   # numeric feature
        age = st.slider("Your age", min_value=10, max_value=80, value=25)
        user_vec.append(age)
    else:
        opts = list(encodings[feat].keys())
        choice = st.selectbox(feat, opts)
        user_vec.append(encodings[feat][choice])

# 4) predict on button click
if st.button("Predict"):
    X = pd.DataFrame([user_vec], columns=features)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]

    st.subheader("Prediction:")
    if pred == "at risk":
        st.error("🚨 This person is likely AT RISK 🚨")
    else:
        st.success("✔️ This person is likely LOW RISK ✔️")



