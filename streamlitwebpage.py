#!/usr/bin/env python
# coding: utf-8

# In[ ]:
#importing the necessary libraries.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def TAS(ax,fontsize=8):
    from collections import namedtuple
    FieldLine = namedtuple('FieldLine', 'x1 y1 x2 y2')
    lines = (FieldLine(x1=41, y1=0, x2=41, y2=7),
             FieldLine(x1=41, y1=7, x2=52.5, y2=14),
             FieldLine(x1=45, y1=0, x2=45, y2=5),
             FieldLine(x1=41, y1=3, x2=45, y2=3),
             FieldLine(x1=45, y1=5, x2=61, y2=13.5),
             FieldLine(x1=45, y1=5, x2=52, y2=5),
             FieldLine(x1=52, y1=5, x2=69, y2=8),
             FieldLine(x1=49.4, y1=7.3, x2=52, y2=5),
             FieldLine(x1=52, y1=5, x2=52, y2=0),
             FieldLine(x1=48.4, y1=11.5, x2=53, y2=9.3),
             FieldLine(x1=53, y1=9.3, x2=57, y2=5.9),
             FieldLine(x1=57, y1=5.9, x2=57, y2=0),
             FieldLine(x1=52.5, y1=14, x2=57.6, y2=11.7),
             FieldLine(x1=57.6, y1=11.7, x2=63, y2=7),
             FieldLine(x1=63, y1=7, x2=63, y2=0),
             FieldLine(x1=69, y1=12, x2=69, y2=8),
             FieldLine(x1=45, y1=9.4, x2=49.4, y2=7.3),
             FieldLine(x1=59.4, y1=10.5, x2=68.8, y2=10.5),
             FieldLine(x1=69, y1=8, x2=77, y2=0)
            )

    FieldName = namedtuple('FieldName', 'name x y rotation')
    names = (FieldName('Picro\nbasalt', 43, 2, 0),
             FieldName('Basalt', 48.5, 2, 0),
             FieldName('Basaltic\nandesite', 54.5, 3.9, 0),
             FieldName('Andesite', 60, 2, 0),
             FieldName('Dacite', 68.5, 2, 0),
             FieldName('Rhyolite', 75.5, 6.5, 0),
             FieldName('Trachyte',
                       64.5, 12.5, 0),
             FieldName('Trachy\ndacite',
                       64.7, 9.5, 0),
             FieldName('2', 52.6, 7.3, 0),
             FieldName('1', 48.7, 6.2, 0),
             FieldName('3', 57.2, 8.8, 0),
             FieldName('Phono\ntephrite', 49, 9.9, 0),
             FieldName('Tephri\nphonolite', 53.0, 12.1, 0),
             FieldName('Phonolite', 57.5, 14.5, 0),
             FieldName('Tephrite', 45, 7.3, 0),
             FieldName('Foidite', 44, 11.5, 0),
             FieldName('Basa\nnite', 43, 4.5, 0))



    for line in lines:
        ax.plot([line.x1, line.x2], [line.y1, line.y2],
                       '-', color='black', zorder=2)
    for name in names:
        ax.text(name.x, name.y, name.name, color='black', size=13,
                    horizontalalignment='center', verticalalignment='top',
                    rotation=name.rotation, zorder=2,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.2')) 


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

test_data_path = "TestFineTuned.csv"  
use_test_data = st.toggle("Use test data")

if use_test_data:
    data = pd.read_csv(test_data_path)
    st.write("Using test data:")
    # st.write(data.head())
else:
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader_key")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.write(data.head())



case = st.radio(
    "Select the case:",
    ['Case 1 - All Oxides', 'Case 2 - No SiO2', 'Case 3 - No Alkali Oxides']
)

# #uploading the test data.
# uploaded_file = st.file_uploader("Upload your CSV file",type=["csv"], key="file_uploader_key")
# if uploaded_file:
#         data = pd.read_csv(uploaded_file)
#         st.write("Uploaded Data:")
#         st.write(data.head())

if 'data' in locals():
    arranged_data = arrange_columns(data, case)
    model = load_model_for_case(case)
    st.write(f"Model for {case} loaded successfully!")
    normalised_data = normalise_data(arranged_data, case)

    if st.button("Predict Rock Type"):
        predictions = model.predict(normalised_data)
        predicted_labels = np.argmax(predictions, axis=1)  
        predicted_rock_types = [label_to_rock[label] for label in predicted_labels]
        arranged_data.insert(0, 'Predicted_Rock_Type', predicted_rock_types)  # Insert at the first column
        st.write(arranged_data)

        csv = arranged_data.to_csv(index=False)
        st.download_button(label="Download Predicted rock type file as CSV",
                           data=csv,
                           file_name='predicted_rock_types.csv',
                           mime='text/csv')

        # TAS Plot
        if case == 'Case 1 - All Oxides':
            fig, ax = plt.subplots(figsize=(8, 6))
            TAS(ax)
            ax.set_xlim([40, 80])
            ax.set_ylim([0, 16])
            ax.set_xlabel('SiO2 (wt%)')
            ax.set_ylabel('Na2O + K2O (wt%)')

            # Plotting the SiO2 vs Na2O+K2O
            ax.scatter(arranged_data['SiO2(wt%)'], arranged_data['Na2O+K2O'], c='red')
            ax.legend()
            st.pyplot(fig)































# if data is not None:
#     arranged_data = arrange_columns(data, case)
#     model = load_model_for_case(case)
#     normalised_data = normalise_data(arranged_data, case)
    
#     if st.button("Predict Rock Type"):
#         predictions = model.predict(normalised_data)
#         predicted_labels = np.argmax(predictions, axis=1)
#         predicted_rock_types = [label_to_rock[label] for label in predicted_labels]

#         # Add the predicted rock type as the first column
#         arranged_data.insert(0, 'Predicted_Rock_Type', predicted_rock_types)

#         st.write(arranged_data)

#         # Download the result as CSV
#         csv = arranged_data.to_csv(index=False)
#         st.download_button(label="Download CSV with predicted rock types", data=csv, file_name='predicted_rock_types.csv', mime='text/csv')

# if uploaded_file:
#     data = pd.read_csv(uploaded_file)
#     st.write("Uploaded Data:")
#     st.write(data.head())
#     arranged_data = arrange_columns(data, case)
#     # st.write("Data with Columns Rearranged:")
#     # st.write(arranged_data.head())
    
# # model_path = 'fine_tuned_model.h5'  
# # model = load_model(model_path, custom_objects={'LeakyReLU': LeakyReLU})

#     model = load_model_for_case(case)
#     st.write(f"Model for {case} loaded successfully!")
#     normalised_data = normalise_data(arranged_data, case)


# if st.button("Predict Rock Type"):
#     predictions = model.predict(normalised_data)
#     predicted_labels = np.argmax(predictions, axis=1)  
#     predicted_rock_types = [label_to_rock[label] for label in predicted_labels]
#     arranged_data.insert(0, 'Predicted_Rock_Type', predicted_rock_types)
#     st.write(arranged_data)

#     csv = data.to_csv(index=False)
#     st.download_button(label="Download Predicted rock type file as csv",
#                            data=csv,
#                            file_name='predicted_rock_types.csv',
#                            mime='text/csv')


















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

