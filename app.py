import streamlit as st
import numpy as np
import pickle

# Load the trained model and scaler
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('D:\BR CAN ml pro 2\scaler (1).pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Web app title
st.title("🧬 Breast Cancer Prediction App")
st.write("Enter the patient's tumor features to predict if it's Benign (B) or Malignant (M).")

# Feature inputs (30 features — make sure they match your dataset)
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

input_values = []
for feature in feature_names:
    value = st.number_input(f"{feature.replace('_', ' ').title()}", min_value=0.0, format="%.4f")
    input_values.append(value)

# Predict button
if st.button("Predict"):
    # Convert input to numpy array and scale
    input_array = np.array(input_values).reshape(1, -1)
    scaled_input = scaler.transform(input_array)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    # Show result
    if prediction == 1:
        st.error("The model predicts: **Malignant (M)**")
    else:
        st.success("The model predicts: **Benign (B)**")
