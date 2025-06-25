import streamlit as st
import pickle as pk
import numpy as np
import os

# ---------- Model File Check ----------
model_path = 'linear_regression_model.pkl'

if not os.path.exists(model_path):
    st.error(f"❌ Model file '{model_path}' not found in the app folder!")
    st.stop()

# ---------- Load the Trained Model ----------
with open(model_path, 'rb') as file:
    model = pk.load(file)

st.title("🏠 Housing Price Predictor")
st.write("This app predicts house prices using a trained Linear Regression model.")

# ---------- Feature Inputs (10 Features Matching Your Model) ----------
st.header("Enter House Features:")

area = st.number_input('Area (in sqft)', min_value=500, max_value=10000, value=1500)

bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=3)

bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2)

stories = st.number_input('Number of Stories', min_value=1, max_value=4, value=1)

mainroad = st.selectbox('Is it on the main road?', ['yes', 'no'])
mainroad_binary = 1 if mainroad == 'yes' else 0

guestroom = st.selectbox('Guest Room Available?', ['yes', 'no'])
guestroom_binary = 1 if guestroom == 'yes' else 0

basement = st.selectbox('Has Basement?', ['yes', 'no'])
basement_binary = 1 if basement == 'yes' else 0

hotwaterheating = st.selectbox('Hot Water Heating?', ['yes', 'no'])
hotwaterheating_binary = 1 if hotwaterheating == 'yes' else 0

airconditioning = st.selectbox('Air Conditioning?', ['yes', 'no'])
airconditioning_binary = 1 if airconditioning == 'yes' else 0

parking = st.slider('Parking Spaces', min_value=0, max_value=5, value=1)

# ---------- Prediction ----------
if st.button('Predict Price'):
    input_data = np.array([
        area,
        bedrooms,
        bathrooms,
        stories,
        mainroad_binary,
        guestroom_binary,
        basement_binary,
        hotwaterheating_binary,
        airconditioning_binary,
        parking
    ]).reshape(1, -1)  # ✅ Correctly closed the list and reshaped

    prediction = model.predict(input_data)[0]  # Get single predicted value
    st.success(f"🏡 Estimated House Price: ₹{round(float(prediction), 2)}")
