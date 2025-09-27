import streamlit as st
import pandas as pd
import numpy as np
import joblib
#To ué this, in cmd run streamlit run test.py
# Load saved objects
model = joblib.load('model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
label_encoder = joblib.load('label_encoder.pkl')

st.title("Weather Condition Predictor")

# Numeric input features
numeric_fields = [
    'tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike',
    'dew', 'humidity', 'precip', 'precipprob', 'precipcover',
    'windgust', 'windspeed', 'sealevelpressure', 'cloudcover',
    'visibility', 'solarradiation', 'solarenergy', 'uvindex'
]

# Categorical input
preciptype = st.selectbox("Precipitation Type", options=["rain", "no_rain"]) 

# Collect numeric inputs
numeric_values = {}
for field in numeric_fields:
    numeric_values[field] = st.number_input(f"{field}", value=0.0)

# When user clicks Predict
if st.button("Predict"):
    # Build a single-row DataFrame for the input
    input_data = {**numeric_values, 'preciptype': preciptype}
    input_df = pd.DataFrame([input_data])

    # Transform using saved preprocessor
    X_transformed = preprocessor.transform(input_df)

    # Predict
    pred_encoded = model.predict(X_transformed)
    pred_label = label_encoder.inverse_transform(pred_encoded)[0]

    st.success(f"Predicted Condition: {pred_label}")
