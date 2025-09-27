import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib

# Load saved model and preprocessor
model = joblib.load('model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
label_encoder = joblib.load('label_encoder.pkl')

st.title("Live Weather Condition Predictor for Hà Nội")

# Visual Crossing API settings
API_KEY = "A3WBTUV48VGFLQBLSQH2RRHMF"
location = "Ha noi"
url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}?unitGroup=us&include=current,days&key={API_KEY}&contentType=json"

if st.button("Predict Using Live Weather"):
    try:
        response = requests.get(url)
        data = response.json()

        # Extract current and day-level data
        current = data['currentConditions']
        day_data = data['days'][0]

        # Build feature input
        input_data = {
            'tempmax': day_data.get('tempmax', current.get('temp', 0)),
            'tempmin': day_data.get('tempmin', current.get('temp', 0)),
            'temp': current.get('temp', 0),
            'feelslikemax': day_data.get('feelslikemax', current.get('feelslike', 0)),
            'feelslikemin': day_data.get('feelslikemin', current.get('feelslike', 0)),
            'feelslike': current.get('feelslike', 0),
            'dew': current.get('dew', 0),
            'humidity': current.get('humidity', 0),
            'precip': current.get('precip', 0),
            'precipprob': day_data.get('precipprob', 0),
            'precipcover': day_data.get('precipcover', 0),
            'windgust': current.get('windgust', 0),
            'windspeed': current.get('windspeed', 0),
            'sealevelpressure': current.get('pressure', 0),
            'cloudcover': current.get('cloudcover', 0),
            'visibility': current.get('visibility', 0),
            'solarradiation': current.get('solarradiation', 0),
            'solarenergy': current.get('solarenergy', 0),
            'uvindex': current.get('uvindex', 0),
        }

        # Extract preciptype correctly
        preciptype_list = day_data.get('preciptype', [])
        if isinstance(preciptype_list, list) and len(preciptype_list) > 0:
            input_data['preciptype'] = preciptype_list[0].lower()
        else:
            input_data['preciptype'] = 'none'

        # Create DataFrame and fill missing values
        input_df = pd.DataFrame([input_data]).fillna(0)

        # Transform input
        X_transformed = preprocessor.transform(input_df)

        # Predict and decode label
        pred_encoded = model.predict(X_transformed)
        pred_label = label_encoder.inverse_transform(pred_encoded)[0]

        st.success(f"🌤️ Predicted Weather Condition: {pred_label}")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
