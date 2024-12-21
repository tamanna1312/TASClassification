import streamlit as st

# st.header('Welcome')
st.header('Classfication of volcanic rocks using ML')
st.write('Data Upload:')
st.write('Users can upload their geochemical dataset in CSV format. The app validates the uploaded file to ensure compatibility with the selected case.')
st.write('Automatic Column Verification:')
st.write('The app checks whether the uploaded data contains all the required element oxides for the selected case (e.g., all oxides for Case 1, excluding SiO₂ for Case 2, or excluding alkali oxides for Case 3).')
st.write('Missing or incorrect columns will prompt the user to correct the dataset.')
st.write('The data is automatically rearranged to match the column order of the model’s training data, ensuring seamless integration.')
st.write('Predicted Rock Type:')
st.write('Once the data passes validation, the app uses the trained machine learning model to classify the rock types.')
st.write('The predicted rock type is displayed in a new column alongside the original dataset in a table format.')
st.write('TAS Plot Visualization:')
st.write('The predicted rock types are simultaneously plotted on the TAS diagram.')
st.write('The TAS plot dynamically updates based on the selected case and uploaded data.')


# st.write('An application designed for classification of volcanic rocks using ML.')

st.session_state.all_data = None
