import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="siddhesh1981/Predictive-Maintenance-Model", filename="bagging_predict_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Purchase Prediction
st.title("Predictive Maintenance Prediction App")
st.write("The Predictive Maintenance Prediction App is an internal tool for Fleet owners and Vehicle Manufacturers, that predicts  whether a Vehicle engine is faulty and requires maintenance or not.")
st.write("Kindly enter the Vehicle engine sensor details to check whether the engine is faulty or not.")

# Collect user input

Engine_rpm=st.number_input('Engine_rpm',min_value=60,max_value=2240,value=746)
Lub_oil_pressure= st.number_input('Lub_oil_pressure',min_value=0.000000,max_value=8.000000,value=3.000000)
Fuel_pressure= st.number_input('Fuel_pressure',min_value=0.000000,max_value=22.000000,value=6.000000)
Coolant_pressure=st.number_input('Coolant_pressure',min_value=0.000000,max_value=8.000000,value=2.000000)
lub_oil_temp=st.number_input('lub_oil_temp',min_value=70.000000,max_value=90.000000,value=76.000000)
Coolant_temp=st.number_input('Coolant_temp',min_value=60.000000,max_value=196.000000,value=78.000000)

input_data = pd.DataFrame([{
    'Engine_rpm': Engine_rpm,
    'Lub_oil_pressure': Lub_oil_pressure,
    'Fuel_pressure': Fuel_pressure,
    'Coolant_pressure': Coolant_pressure,
    'lub_oil_temp': lub_oil_temp,
    'Coolant_temp': Coolant_temp  
                             }])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data).astype(int)    
    result = "Faulty and requires maintenance" if prediction == 1 else "NonFaulty and does not require maintenance"
    st.write(f"Based on the vehicle engine sensor information provided, the vehicle engine is likely to be {result}.")







