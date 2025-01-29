import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Path to the saved model file
model_path = "C:/Users/asus/Desktop/SupportTicket Model/breast_cancer_rf_model.pkl"

# Load the model
st.title("Breast Cancer Prediction")
st.header("Predict the likelihood of breast cancer based on input features")

try:
    loaded_model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Model file not found at: {model_path}. Please check the file path and try again.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading the model: {e}")
    st.stop()

# App description
st.write("""
This app predicts the likelihood of breast cancer using a machine learning model. 
Enter the relevant input features below to get the prediction.
""")

# Create input fields for user to input feature values
st.subheader("Enter the Input Features")
try:
    feature1 = st.number_input("Feature 1 (e.g., radius_mean)", min_value=0.0, step=0.1, format="%.2f")
    feature2 = st.number_input("Feature 2 (e.g., texture_mean)", min_value=0.0, step=0.1, format="%.2f")
    feature3 = st.number_input("Feature 3 (e.g., perimeter_mean)", min_value=0.0, step=0.1, format="%.2f")
    # Extend for additional features as per your model's requirements

    # Collect inputs into a numpy array
    input_features = np.array([[feature1, feature2, feature3]])  # Adjust based on the model's feature count
except Exception as e:
    st.error(f"Error in handling input features: {e}")
    st.stop()

# Predict using the loaded model
if st.button("Predict"):
    try:
        # Make prediction
        prediction = loaded_model.predict(input_features)
        prediction_proba = loaded_model.predict_proba(input_features)

        # Display prediction result
        if prediction[0] == 1:
            st.success("The model predicts: The patient is likely to have breast cancer.")
        else:
            st.success("The model predicts: The patient is unlikely to have breast cancer.")

        # Display prediction probabilities
        st.subheader("Prediction Probabilities")
        st.write(f"Class 0 (No Cancer): {prediction_proba[0][0] * 100:.2f}%")
        st.write(f"Class 1 (Cancer): {prediction_proba[0][1] * 100:.2f}%")

    except ValueError as ve:
        st.error(f"Value error during prediction. Ensure all inputs are valid: {ve}")
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")
