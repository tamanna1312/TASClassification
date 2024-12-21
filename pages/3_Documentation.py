import streamlit as st

# st.title('Documentation for Creating Fake Data')
# st.header('Classfication of volcanic rocks using ML')
st.header('Data Upload:')
st.write('Users can upload their geochemical dataset in CSV format. The app validates the uploaded file to ensure compatibility with the selected case.')
st.header('Automatic Column Verification:')
st.write('The app checks whether the uploaded data contains all the required element oxides for the selected case (e.g., all oxides for Case 1, excluding SiO₂ for Case 2, or excluding alkali oxides for Case 3).Missing or incorrect columns will prompt the user to correct the dataset.')
st.write('The data is automatically rearranged to match the column order of the model’s training data.')
st.header('Predicted Rock Type:')
st.write('Once the data passes validation, the app uses the trained ML model to classify the rock types. The predicted rock type is displayed in a new column alongside the original dataset in a table format.')
st.header('TAS Plot Visualization:')
st.write('The predicted rock types are simultaneously plotted on the TAS diagram.')

# The  fake data were generated using a hybrid statistical approach. The mean and covariance matrix for each rock type were computed from the real-world geochemical data, which captured the central tendencies and interdependencies of all the oxide compositions. And as we mentioned, for SiO₂ and Na₂O+K₂O, values were sampled from the vertices of the TAS diagram for the three cases- (i) shared borders (ii) no common points before borders and (iii) 10% away from the borders. The remaining 7 oxides were then created using a multivariate normal distribution, which ensured that the fake data reflected the variability and correlations that are observed in real-world geochemical data. Also, controlled noise was added to introduce slight variations and account for natural geochemical heterogeneity. Finally, the data from these different sources were combined, and the fake dataset was carefully validated to ensure geochemical plausibility. The code for the same is uploaded on the github repository.
# Below, you can also see the correlation matrices for all the oxides for both real and fake data.
