import streamlit as st
st.header(':blue[Using TAS Classifier]')
st.subheader('Introduction')
st.write('The TAS Classifier is a ML based tool designed for automatic rock classification using 10 major and minor oxides. By uploading your dataset, the app predicts rock types and visualizes them on a TAS (Total Alkali and Silica) plot. The classifier supports three distinct cases, allowing users to explore scenarios where certain major oxides (e.g., SiO₂ or alkali oxides) are excluded.')
st.header(':blue[Creating Fake Data]')
st.subheader('Introduction')
st.write('The Fake Data Creation module enables users to create fake geochemical data based on real-world data. Using statistical techniques, such as mean and covariance analysis, the app creates realistic geochemical data that mirrors the variability and correlations observed in real samples. Users can visualize correlation matrices for both real and generated data as well as download the fake data.')















# st.write('Users can upload their geochemical dataset in CSV format. The app validates the uploaded file to ensure compatibility with the selected case.')
# st.subheader('Automatic column verification:')
# st.write('The app checks whether the uploaded data contains all the required element oxides for the selected case (e.g., all oxides for Case 1, excluding SiO₂ for Case 2, or excluding alkali oxides for Case 3). Missing or incorrect columns will prompt the user to correct the dataset. The data is automatically rearranged to match the column order of the model’s training data.')
# st.subheader('Predicted rock type')
# st.write('Once the data passes validation, the app uses the trained ML model to classify the rock types. The predicted rock type is displayed in a new column alongside the original dataset in a table format.')
# st.subheader('TAS plot visualization')
# st.write('The predicted rock types are simultaneously plotted on the TAS diagram.')

# st.subheader('Upload real-world data')
# st.write('You can upload real-world geochemical data in CSV format')
# st.subheader('How fake data is created')
# st.write('Statistical Basis: The mean and covariance matrix for each rock type are computed from the real-world data. This captures the central tendencies and interdependencies among all oxide compositions. For SiO₂ and Na₂O+K₂O, values are sampled based on specific TAS plot vertices. The remaining 7 oxides are created using a multivariate normal distribution. This approach ensures that the generated data reflects real-world variability and correlations.')
# st.subheader('View correlation matrices and download fake data')
# st.write('Once the fake data is created, the app computes and displays correlation matrices for both the real-world and fake datasets. These matrices help visualize the relationships between oxides and ensure that the simulated data preserves the statistical characteristics of the real data. The fake data can then be downloaded as a CSV file.')
