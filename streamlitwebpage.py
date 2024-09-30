#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU  
from sklearn.preprocessing import StandardScaler


st.title('TAS Rock Classifier')

def load_model_for_case(case):
    if case == 'Case 1 - All Oxides':
        return load_model('fine_tuned_model.h5')
    elif case == 'Case 2 - No SiO2':
        return load_model('model_case_2.h5')
    elif case == 'Case 3 - No Alkali Oxides':
        return load_model('model_case_3.h5')


def normalize_data(data, case):
    scaler = StandardScaler()
    # Depending on the case, exclude certain columns for normalization
    if case == 'Case 1 - All Oxides':
        features = data  
    elif case == 'Case 2 - No SiO2':
        features = data.drop(columns=['SiO2'])
    elif case == 'Case 3 - No Alkali Oxides':
        features = data.drop(columns=['Na2O', 'K2O'])


    normalized_data = scaler.fit_transform(features)
    return normalized_data

def arrange_columns(data, case):
    if case == 'Case 1 - All Oxides':
        # Specify the exact column order for Case 1
        column_order = ['SiO2(wt%)', 'TiO2(wt%)', 'Al2O3(wt%)', 'FeOT(wt%)', 'CaO(wt%)', 'MgO(wt%)',  'MnO(wt%)', 'P2O5(wt%)', 'Na2O+K2O', 'Na2O+K2O/SiO2']
    elif case == 'Case 2 - No SiO2':
        # Specify the exact column order for Case 2
        column_order = ['Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O', 'TiO2', 'MnO', 'P2O5']
    elif case == 'Case 3 - No Alkali Oxides':
        # Specify the exact column order for Case 3
        column_order = ['SiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'TiO2', 'MnO', 'P2O5']
    
    return data[column_order]


case = st.selectbox(
    "Select the case:",
    ['Case 1 - All Oxides', 'Case 2 - No SiO2', 'Case 3 - No Alkali Oxides']
)

# Step 2: Upload the CSV file with user data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(data.head())

    # Rearrange the columns based on the model's training data
    arranged_data = arrange_columns(data, case)

    st.write("Data with Columns Rearranged:")
    st.write(arranged_data.head())
    
model_path = 'fine_tuned_model.h5'  # Path to the pre-saved model file
model = load_model(model_path, custom_objects={'LeakyReLU': LeakyReLU})

st.write("Model loaded from disk!")

    # # Load the model based on selected case
    # model = load_model_for_case(case)

# Normalize the data based on the selected case
normalized_data = normalize_data(arranged_data, case)

# Step 3: Make predictions
if st.button("Predict Rock Type"):
    predictions = model.predict(normalized_data)
    predicted_labels = np.argmax(predictions, axis=1)  # Assuming a classification task with multiple classes
    data['Predicted_Rock_Type'] = predicted_labels
    st.write("Predictions:")
    st.write(data)

# Step 4: Allow the user to download the new CSV file
    csv = data.to_csv(index=False)
    st.download_button(label="Download Predicted Data as CSV",
                           data=csv,
                           file_name='predicted_rock_types.csv',
                           mime='text/csv')

# # @st.cache(allow_output_mutation=True)
# # def load_results():
# #     # Load the precomputed predictions
# #     results = pd.read_csv('predicted_results.csv')
# #     return results

# # results = load_results()

# # Button to show results
# if st.button('Upload Data'):
#     # Display the DataFrame with the inverse-transformed features and predictions
#     st.write("Data with Predicted Rock Types")
#     st.dataframe(results)
# else:
#     st.write("Click the button to upload your data.")

