import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Define paths to the transformer and model pickle files
transformer_path = 'saved_models/1/transformer/transformer.pkl'
model_path = 'saved_models/1/model/model.pkl'
data_path = 'data/KDDTrain_.csv'

# Load transformer and model objects from pickle files
transformer = pkl.load(open(transformer_path, "rb"))
model = pkl.load(open(model_path, "rb"))

st.title("KDD PREDICTION")

# Load original data
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

# Prepare data for prediction
original_data_sample = original_data.head(30)  # Take only the first 10 rows for prediction
predictions = predict(original_data_sample)

# Create a new DataFrame with original data and predicted values
prediction_data = original_data_sample.copy()
prediction_data['Predicted'] = ['Anomaly' if pred == 1 else 'Normal' for pred in predictions]

st.subheader("Data with Predicted Values")
st.write(prediction_data)
