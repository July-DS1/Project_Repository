import streamlit as st
import joblib
import numpy as np

st.title("🔧 Machine Failure Prediction App")

st.write("Enter machine parameters to predict failure risk.")

model = joblib.load("model/model.pkl")

air_temp = st.number_input("Air temperature (K)", value=300)
process_temp = st.number_input("Process temperature (K)", value=310)
speed = st.number_input("Rotational speed (rpm)", value=1500)
torque = st.number_input("Torque (Nm)", value=40)
tool_wear = st.number_input("Tool wear (minutes)", value=120)

if st.button("Predict Failure"):

    data = np.array([[air_temp, process_temp, speed, torque, tool_wear]])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        st.error(f"⚠ Machine Failure Risk! Probability: {probability:.2f}")
    else:
        st.success(f"✅ Machine Operating Normally. Probability: {probability:.2f}")