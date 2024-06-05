#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the trained model and the scaler
model = load_model('TASClassification.h5')
sc = joblib.load('scaler.pkl')

label_to_rock = {0: 'Andesite', 1: 'Basalt', 2: 'Basaltic Andesite', 3: 'Basaltic Trachyandesite', 4: 'Basanite', 5:'Dacite', 6: 'Foidite', 7: 'Phonolite',
                 8: 'Phonotephrite', 9: 'Picrobasalt', 10: 'Rhyolite', 11: 'Tephriphonolite', 12: 'Tephrite', 13: 'Trachyandesite',
                 14: 'Trachybasalt', 15: 'Trachydacite', 16: 'Trachyte'
                }  

st.title('Geochemical Data Rock Label Predictor')

# Function to load and preprocess the uploaded data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    feature_names = data.columns
    features = sc.transform(data)
    return data, features, feature_names

# Function to make predictions
def make_predictions(features):
    predictions = model.predict(features)
    predicted_labels = np.argmax(predictions, axis=1)
    return predictions, predicted_labels

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load and preprocess the data
    original_data, scaled_features, feature_names = load_data(uploaded_file)

    # Make predictions
    predictions, predicted_labels = make_predictions(scaled_features)
    
    # Convert numeric labels to rock names
    predicted_rock_names = [label_to_rock[label] for label in predicted_labels]
    
    # Revert normalization
    X_original = sc.inverse_transform(scaled_features)
    
    # Create a DataFrame with the original features and predicted labels
    df_original = pd.DataFrame(X_original, columns=feature_names)
    df_original['Predicted_Label'] = predicted_rock_names
    
    st.write("### Original Data with Predicted Labels")
    st.write(df_original)
    
    # Provide download link for the data with predictions
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df_original)
    st.download_button(
        label="Download data with predictions",
        data=csv,
        file_name='predicted_labels.csv',
        mime='text/csv',
    )

