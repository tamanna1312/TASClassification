#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import pandas as pd

st.title('TAS Rock Classifier')

@st.cache(allow_output_mutation=True)
def load_results():
    # Load the precomputed predictions
    results = pd.read_csv('predicted_results.csv')
    return results

results = load_results()

# Button to show results
if st.button('Upload Data'):
    # Display the DataFrame with the inverse-transformed features and predictions
    st.write("Data with Predicted Rock Types")
    st.dataframe(results)
else:
    st.write("Click the button to display the predictions.")

