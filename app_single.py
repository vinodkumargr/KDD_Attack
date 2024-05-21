import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl

# Define paths to the transformer and model pickle files
transformer_path = 'saved_models/1/transformer/transformer.pkl'
model_path = 'saved_models/1/model/model.pkl'

# Load transformer and model objects from pickle files
transformer = pkl.load(open(transformer_path, "rb"))
model = pkl.load(open(model_path, "rb"))

st.title("KDD PREDICTION")

# Load original data
data_path = 'data/KDDTrain_.csv'
original_data = pd.read_csv(data_path)

st.subheader("Original Data (First 10 rows)")
st.write(original_data.head(30))

# Define function to preprocess input data
def preprocess_input(input_data):
    # Perform any necessary preprocessing here
    preprocessed_data = input_data  # Placeholder for actual preprocessing

    # Transform input data using the loaded transformer
    transformed_input = transformer.transform(preprocessed_data)

    return transformed_input

# Define function to make predictions
def predict(input_data):
    # Preprocess input data
    preprocessed_input = preprocess_input(input_data)

    # Make predictions using the loaded model
    predictions = model.predict(preprocessed_input)

    return predictions

# Prepare data for single prediction
st.subheader("Input Features for Single Prediction")
input_data = {}

# Generate input fields dynamically for all features except the target column 'class'
for column_name in original_data.columns:
    if column_name != 'class':
        if np.issubdtype(original_data[column_name].dtype, np.number):  # For numerical columns
            input_data[column_name] = st.number_input(f"{column_name}", key=f"{column_name}")
        else:  # For categorical columns
            unique_values = original_data[column_name].unique()
            input_data[column_name] = st.selectbox(f"{column_name}", unique_values, key=f"{column_name}")

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Make single prediction when the button is clicked
if st.button('Predict'):
    prediction = predict(input_df)
    result = 'Anomaly' if prediction[0] == 1 else 'Normal'
    st.write(f"Predicted result: {result}")
