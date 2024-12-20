import streamlit as st

st.title('Documentation')
# In short, the  fake data were generated using a hybrid statistical approach. The mean and covariance matrix for each rock type were computed from the real-world geochemical data, which captured the central tendencies and interdependencies of all the oxide compositions. And as we mentioned, for SiO₂ and Na₂O+K₂O, values were sampled from the vertices of the TAS diagram for the three cases- (i) shared borders (ii) no common points before borders and (iii) 10% away from the borders. The remaining 7 oxides were then created using a multivariate normal distribution, which ensured that the fake data reflected the variability and correlations that are observed in real-world geochemical data. Also, controlled noise was added to introduce slight variations and account for natural geochemical heterogeneity. Finally, the data from these different sources were combined, and the fake dataset was carefully validated to ensure geochemical plausibility. The code for the same is uploaded on the github repository.
# Below, you can also see the correlation matrices for all the oxides for both real and fake data.
