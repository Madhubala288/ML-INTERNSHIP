import streamlit as st
import joblib
import numpy as np
model = joblib.load('models/diabetes_model.pkl')
scaler = joblib.load('models/scaler.pkl')
st.title("Diabetes Prediction System")
preg = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose", min_value=0.0)
bp = st.number_input("Blood Pressure", min_value=0.0)
skin = st.number_input("Skin Thickness", min_value=0.0)
insulin = st.number_input("Insulin", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0, step=1)
if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    if prediction[0] == 0:
        st.success("Person is Non-Diabetic")
    else:
        st.error("Person is Diabetic")