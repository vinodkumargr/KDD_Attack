# KKD Attack Prediction

KKD Attack Prediction is a machine learning project focused on predicting two types of network traffic: normal and anomaly. The project includes data visualization, model comparison, and an interactive user interface for predictions using Streamlit.

## Table of Contents
- [Overview](#overview)
- [Data Visualization](#data-visualization)
- [Model Comparison](#model-comparison)
- [Final Model](#final-model)
- [Streamlit UI](#streamlit-ui)
- [Installation](#installation)
- [Usage](#usage)

## Overview
This project aims to accurately predict whether network traffic is normal or anomalous. By comparing various classification algorithms, we have determined that Logistic Regression provides the best performance for our dataset.

## Data Visualization
We conducted extensive data visualization to understand the characteristics of our dataset. The visualizations helped in identifying patterns and anomalies, which guided the feature selection and preprocessing steps.

## Model Comparison
We compared the performance of three machine learning algorithms:
- **K-Nearest Neighbors (KNN) Classifier**
- **Decision Tree Classifier**
- **Logistic Regression**

### Evaluation Metrics
To evaluate the models, we used the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score

### Results
Among the algorithms tested, Logistic Regression emerged as the best performer with an accuracy of 96%.

## Final Model
The Logistic Regression model was chosen as the final algorithm due to its superior performance. The model achieved:
- **Accuracy:** 96%
- **Precision:** High
- **Recall:** High
- **F1 Score:** High

The model was trained and evaluated using standard classification metrics to ensure robust performance.

## Streamlit UI
We implemented a user-friendly interface using Streamlit for both single and batch predictions. The UI allows users to input data and get real-time predictions, making it easy to use even for non-technical users.

## Installation
To run this project locally, follow these steps:

1. **Clone the repository:**
   ```
   git clone https://github.com/yourusername/kkd-attack-prediction.git
   cd kkd-attack-prediction
   ```

2. **Install the required dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Run the main project:**
   ```
   python main.py
   ```

4. **Run the to save the batch predictions file locally:**
   ```
   python demo.py
   ```
   
4. **Run the Streamlit app for single predictions:**
   ```
   streamlit run app_single.py
   ```

5. **Run the Streamlit app fir batch predictions:**
   ```
   streamlit run app.py
   ```


## Usage
### Single Prediction
1. Open the Streamlit app.
2. Enter the feature values for the prediction.
3. Click on the "Predict" button to get the result.

### Batch Prediction
1. Open the Streamlit app.
2. You can see the actual and predicted values.
