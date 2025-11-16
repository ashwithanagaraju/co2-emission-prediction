

import joblib

import gradio as gr

# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# import numpy as np
# import pandas as pd
# Assuming we're using a model like from scikit-learn, TensorFlow, or any other library

import pickle
import pandas as pd

# This is just an example, adjust import according to your model type

def load_model():
    # Placeholder for loading your predictive model (e.g., a trained machine learning model)
    # Replace 'path/to/your/model' with the actual path to your model file
    model = joblib.load(r'C:\Users\ASHWITHA PN\Downloads\co2emission\co2emission\random_forest_model.pkl')


    return model

def preprocess_input(make, model, vehicle_class, transmission, fuel_type, engine_size, cylinders, fuel_consumption_city, fuel_consumption_hwy, fuel_consumption_comb, fuel_consumption_comb_mpg):
    # Prepare the input data in the form the model expects
    # This might include converting categories to one-hot encoding, scaling/normalizing numerical values, etc.
    # Here we create a DataFrame as an example; adjust according to how your model was trained
    preprocessor=joblib.load(r'C:\Users\ASHWITHA PN\Downloads\co2emission\co2emission\preprocessor.pkl')
    input_data = pd.DataFrame({
        'Make': [make],
        'Model': [model],
        'Vehicle Class': [vehicle_class],
        'Transmission': [transmission],
        'Fuel Type': [fuel_type],
        'Engine Size(L)': [engine_size],
        'Cylinders': [cylinders],
        'Fuel Consumption City (L/100 km)': [fuel_consumption_city],
        'Fuel Consumption Hwy (L/100 km)': [fuel_consumption_hwy],
        'Fuel Consumption Comb (L/100 km)': [fuel_consumption_comb],
        'Fuel Consumption Comb (mpg)':[fuel_consumption_comb_mpg]

    })

    input_data['Engine Size to Cylinders Ratio'] = input_data['Engine Size(L)'] / input_data['Cylinders']
    input_data['Fuel Efficiency Difference'] = input_data['Fuel Consumption Hwy (L/100 km)'] - input_data['Fuel Consumption City (L/100 km)']

    data = preprocessor.transform(input_data)


    return data

def pr(make, model, vehicle_class, transmission, fuel_type, engine_size, cylinders, fuel_consumption_city, fuel_consumption_hwy, fuel_consumption_comb, fuel_consumption_comb_mpg):
    # Load model
    model = load_model()

    # Preprocess inputs
    preprocessed_input = preprocess_input(make, model, vehicle_class, transmission, fuel_type, engine_size, cylinders, fuel_consumption_city, fuel_consumption_hwy, fuel_consumption_comb, fuel_consumption_comb_mpg)
    print(preprocessed_input)

    # Predict using the model
    prediction = model.predict(preprocessed_input)
    return  f"Predicted CO2 Emissions: {prediction[0]:.2f} g/km"

result =pr('BMW', 'MDX 4WD', 'SUV - SMALL', 'AS8', 'D', 3.0, 6, 9.5, 7.2, 8.4, 28)


# Defining the interface
iface = gr.Interface(
    # Function to be called when inputs are provided
    fn=pr,
     # Defining inputs for the interface
    inputs=[
        gr.Dropdown(choices=['ACURA', 'ALFA ROMEO', 'ASTON MARTIN', 'AUDI', 'BENTLEY', 'BMW', 'BUICK',
 'CADILLAC', 'CHEVROLET', 'CHRYSLER', 'DODGE', 'FIAT', 'FORD', 'GMC', 'HONDA',
 'HYUNDAI', 'INFINITI', 'JAGUAR', 'JEEP', 'KIA', 'LAMBORGHINI', 'LAND ROVER',
 'LEXUS', 'LINCOLN', 'MASERATI', 'MAZDA', 'MERCEDES-BENZ', 'MINI', 'MITSUBISHI',
 'NISSAN', 'PORSCHE', 'RAM', 'ROLLS-ROYCE', 'SCION', 'SMART', 'SRT', 'SUBARU',
 'TOYOTA', 'VOLKSWAGEN', 'VOLVO', 'GENESIS', 'BUGATTI'], label="Make"),
        gr.Dropdown(choices=['ILX', 'ILX HYBRID', 'MDX 4WD', 'Tacoma 4WD D-Cab TRD Off-Road/Pro',
 'Atlas Cross Sport 4MOTION', 'XC40 T4 AWD'], label="Model"),
        gr.Dropdown(choices=['COMPACT', 'SUV - SMALL', 'MID-SIZE', 'TWO-SEATER', 'MINICOMPACT',
 'SUBCOMPACT', 'FULL-SIZE', 'STATION WAGON - SMALL', 'SUV - STANDARD',
 'VAN - CARGO', 'VAN - PASSENGER', 'PICKUP TRUCK - STANDARD', 'MINIVAN',
 'SPECIAL PURPOSE VEHICLE', 'STATION WAGON - MID-SIZE',
 'PICKUP TRUCK - SMALL'], label="Vehicle Class"),
        gr.Dropdown(choices=['AS5', 'M6', 'AV7', 'AS6', 'AM6', 'A6', 'AM7', 'AV8', 'AS8', 'A7', 'A8', 'M7', 'A4',
 'M5', 'AV', 'A5', 'AS7', 'A9', 'AS9', 'AV6', 'AS4', 'AM5', 'AM8', 'AM9', 'AS10',
 'A10', 'AV10'], label="Transmission"),
        gr.Dropdown(choices=['Z', 'D', 'X', 'E', 'N'], label="Fuel Type"),
        gr.Number(label="Engine Size (L) - Range 2-10"),
        gr.Number(label="Cylinders - Integers in range 3-16"),
        gr.Number(label="Fuel Consumption City (L/100 km)"),
        gr.Number(label="Fuel Consumption Hwy (L/100 km)"),
        gr.Number(label="Fuel Consumption Comb (L/100 km)"),
        gr.Number(label="Fuel Consumption Comb (mpg)")
    ],
    outputs="text",
    title="CO2 EMISSION Prediction Random Forest Model",
    description="Select options from the dropdowns and enter numerical values in the other inputs to predict the CO2 emission .",
    theme="default"
)
iface.launch(share=True, debug=True)



