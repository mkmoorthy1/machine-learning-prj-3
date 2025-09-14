import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
train_columns = joblib.load("columns.pkl")

st.title("Car Price Prediction App 2")

model_name = st.selectbox(
    "Car Model", ["Tourneo Connect", "Fiesta", "Focus", "Kuga", "S-MAX"])
year = st.number_input("Year", min_value=1996, max_value=2060, value=2019)

transmission = st.selectbox("Transmission",
                            ["Manual", "Automatic", "Semi-Auto"])

mileage = st.number_input("Mileage",
                          min_value=1,
                          max_value=13000,
                          value=5000,
                          step=100)
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid"])
tax = st.number_input("Tax", min_value=0.0, max_value=500.0, value=100.0)
mpg = st.number_input("Miles per Gallon (MPG)",
                      min_value=20.8,
                      max_value=200.0,
                      value=145.0)
engine_size = st.number_input("Engine Size",
                              min_value=1.0,
                              max_value=4.9,
                              value=1.5)

cat_cols = ['model', 'transmission', 'fuelType']
num_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']

input_data = pd.DataFrame([{
    "model": model_name,
    "year": year,
    "transmission": transmission,
    "mileage": mileage,
    "fuelType": fuel,
    "tax": tax,
    "mpg": mpg,
    "engineSize": engine_size
}])

input_data[num_cols] = scaler.transform(input_data[num_cols])
input_data = pd.get_dummies(input_data, drop_first=True)

input_data = input_data.reindex(columns=train_columns, fill_value=0)
prediction = model.predict(input_data)

if st.button("Predict"):
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")
