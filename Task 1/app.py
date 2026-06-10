import streamlit as st
import pickle
import numpy as np
# Page configuration
st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("Student Performance Predictor")
st.write("Enter the student's details below to get a performance prediction.")
# Input Fields
study_hours = st.number_input("Weekly Self Study Hours", min_value=0.0, step=0.5)
attendance = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, step=1.0)
participation = st.number_input("Class Participation Score", min_value=0.0, step=1.0)
previous_score = st.number_input("Previous Test Score", min_value=0.0, step=1.0)
# Model selection
model_type = st.selectbox("Choose Prediction Type", ["Grade Prediction", "Score Prediction"])
if st.button("Predict"):
    try:
        # 1. Prepare input data
        input_data = np.array([[study_hours, attendance, participation, previous_score]])
        # 2. Load Scaler and transform data
        # This ensures inputs match the scale used during training
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        input_data_scaled = scaler.transform(input_data)
        if model_type == "Grade Prediction":
            # Using Decision Tree Modelstreamlit run app.py
            model = pickle.load(open('Decision_Tree_model.pkl', 'rb'))
            prediction = model.predict(input_data_scaled)
            st.success(f"The Predicted Grade is: **{prediction[0]}**")     
        else:
            # Using Logistic Regression Model
            model = pickle.load(open('Logistic_Regression_model.pkl', 'rb'))
            prediction = model.predict(input_data_scaled)
            # Formatting the result to 2 decimal places
            st.success(f"The Predicted Score is: **{float(prediction[0]):.2f}**")
    except FileNotFoundError:
        st.error("Error: Model or Scaler files not found. Please ensure .pkl files are in the directory.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")