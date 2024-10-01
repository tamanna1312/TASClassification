#!/usr/bin/env python
# coding: utf-8

# In[ ]:
#importing the necessary libraries.
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU  
from sklearn.preprocessing import StandardScaler

#title of the app.
st.title('TAS Rock Classifier')

#loading the model for the 3 cases.
def load_model_for_case(case):
    if case == 'Case 1 - All Oxides':
        return load_model('fine_tuned_model.h5', custom_objects={'LeakyReLU': LeakyReLU})
    elif case == 'Case 2 - No SiO2':
        return load_model('model_case_2.h5', custom_objects={'LeakyReLU': LeakyReLU})
    elif case == 'Case 3 - No Alkali Oxides':
        return load_model('model_case_3.h5', custom_objects={'LeakyReLU': LeakyReLU})


# def normalize_data(data, case):
#     scaler = StandardScaler()
#     # Depending on the case, exclude certain columns for normalization
#     if case == 'Case 1 - All Oxides':
#         features = data  
#     elif case == 'Case 2 - No SiO2':
#         features = data.drop(columns=['SiO2'])
#     elif case == 'Case 3 - No Alkali Oxides':
#         features = data.drop(columns=['Na2O', 'K2O'])
#     normalized_data = scaler.fit_transform(features)
#     return normalized_data

#normalising the data.
def normalise_data(data, case):
    scaler = StandardScaler()
    features = data
    normalised_data = scaler.fit_transform(features)
    return normalised_data
    
#arranging the columns of the test data in the same way as that of training data.
def arrange_columns(data, case):
    if case == 'Case 1 - All Oxides':
        column_order = ['SiO2(wt%)', 'TiO2(wt%)', 'Al2O3(wt%)', 'FeOT(wt%)', 'CaO(wt%)', 'MgO(wt%)',  'MnO(wt%)', 'P2O5(wt%)', 'Na2O+K2O', 'Na2O+K2O/SiO2']
    elif case == 'Case 2 - No SiO2':
        column_order = ['TiO2(wt%)', 'Al2O3(wt%)', 'FeOT(wt%)', 'CaO(wt%)', 'MgO(wt%)',  'MnO(wt%)', 'P2O5(wt%)', 'Na2O+K2O', 'Na2O+K2O/SiO2']
    elif case == 'Case 3 - No Alkali Oxides':
        column_order = ['SiO2(wt%)', 'TiO2(wt%)', 'Al2O3(wt%)', 'FeOT(wt%)', 'CaO(wt%)', 'MgO(wt%)',  'MnO(wt%)', 'P2O5(wt%)']
    
    return data[column_order]


#mapping key.
label_to_rock = {0: 'Andesite', 1: 'Basalt', 2: 'Basaltic Andesite', 3: 'Basanite', 4:'Dacite',
                 5: 'Foidite', 6: 'Phonolite', 7: 'Phonotephrite', 8: 'Picrobasalt', 
                 9: 'Rhyolite', 10: 'Tephrite', 11: 'Trachyandesite', 12: 'Trachybasalt', 
                 13: 'Trachydacite', 14: 'Trachyte'}

data_option = st.radio(
    "Select the data source:",
    ('Upload your data', 'Use test data')
)

if data_option == 'Upload your data':
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
else:
    data = pd.read_csv(TestFineTuned.csv)
    st.write("Using test data:")
    st.write(data.head())



case = st.radio(
    "Select the case:",
    ['Case 1 - All Oxides', 'Case 2 - No SiO2', 'Case 3 - No Alkali Oxides']
)

#uploading the test data.
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(data.head())
    arranged_data = arrange_columns(data, case)
    # st.write("Data with Columns Rearranged:")
    # st.write(arranged_data.head())
    
# model_path = 'fine_tuned_model.h5'  
# model = load_model(model_path, custom_objects={'LeakyReLU': LeakyReLU})

    model = load_model_for_case(case)
    st.write(f"Model for {case} loaded successfully!")
    normalised_data = normalise_data(arranged_data, case)


if st.button("Predict Rock Type"):
    predictions = model.predict(normalised_data)
    predicted_labels = np.argmax(predictions, axis=1)  
    predicted_rock_types = [label_to_rock[label] for label in predicted_labels]
    arranged_data.insert(0, 'Predicted_Rock_Type', predicted_rock_types)
    st.write(arranged_data)

    csv = data.to_csv(index=False)
    st.download_button(label="Download Predicted rock type file as csv",
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

