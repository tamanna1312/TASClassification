#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import io


st.title('TAS Rock Classifier')

def load_model_for_case(case):
    if case == 'Case 1 - All Oxides':
        return load_model('model_case_1.h5')
    elif case == 'Case 2 - No SiO2':
        return load_model('model_case_2.h5')
    elif case == 'Case 3 - No Alkali Oxides':
        return load_model('model_case_3.h5')

# @st.cache(allow_output_mutation=True)
# def load_results():
#     # Load the precomputed predictions
#     results = pd.read_csv('predicted_results.csv')
#     return results

results = load_results()

# Button to show results
if st.button('Upload Data'):
    # Display the DataFrame with the inverse-transformed features and predictions
    st.write("Data with Predicted Rock Types")
    st.dataframe(results)
else:
    st.write("Click the button to upload your data.")

