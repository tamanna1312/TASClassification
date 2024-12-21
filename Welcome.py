import streamlit as st

# st.header('Welcome')
st.header('Welcome')
st.write('The app contains 2 sections:')
st.write(':purple[Classify Volcanic Rocks using ML model'])
Leverage a powerful machine learning model to classify volcanic rocks based on major and minor oxide data. The model predicts rock types from real-world and synthetic data, helping you understand complex rock compositions with ease.
Generate Fake Data from Real-World Samples
Use real-world data to generate synthetic datasets that mimic geochemical characteristics. This feature allows you to create "fake" data, ideal for testing or simulating various geological scenarios.



# st.write('An application designed for classification of volcanic rocks using ML.')

st.session_state.all_data = None
