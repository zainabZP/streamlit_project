

import pickle
import streamlit as st
import numpy as np

# Load the pre-trained diabetes model
with open('model_diabetes_logistic.sav', 'rb') as file:
    diabetes_model = pickle.load(file)

# Set up the web app title
st.title("Diabetes Prediction Application")

# Collect input values through user interface
st.header("Enter the following details:")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1, format="%d")

with col2:
    glucose = st.number_input("Glucose Level", min_value=0.0, step=1.0, format="%.2f")

with col1:
    blood_pressure = st.number_input("Blood Pressure Level", min_value=0.0, step=1.0, format="%.2f")

with col2:
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, step=1.0, format="%.2f")

with col1:
    insulin = st.number_input("Insulin Level", min_value=0.0, step=1.0, format="%.2f")

with col2:
    bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, step=0.1, format="%.2f")

with col1:
    pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01, format="%.4f")

with col2:
    age = st.number_input("Age", min_value=0, step=1, format="%d")

# Predict diabetes based on the input values
if st.button("Check for Diabetes"):
    # Ensure all inputs are valid
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree_function, age]])
    
    prediction = diabetes_model.predict(input_data)
    result = "Diabetes Detected" if prediction[0] == 1 else "No Diabetes Detected"

    st.success(result)
