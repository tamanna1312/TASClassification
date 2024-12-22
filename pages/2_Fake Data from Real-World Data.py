#!/usr/bin/env python
# coding: utf-8

# In[ ]:
#importing the necessary libraries.
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# App title
st.title("Fake Data Generator and Correlation Analysis")

# Upload real data
# st.header("Upload Real Data")
uploaded_file = st.file_uploader("Upload a CSV file containing the real data", type="csv")

if uploaded_file:

    data_filtered = pd.read_csv(uploaded_file)
    st.write("Preview of Real Data:")
    st.dataframe(data_filtered.head())

    # Define rock type column and exclude it from oxide columns
    rock_type_column = 'Rock_name'
    oxide_columns = [col for col in data_filtered.columns if col != rock_type_column]

    # Button to generate fake data
    if st.button("Produce Fake Data"):
        # Generate fake data
        n_samples = 100
        fake_data = []

        for rock_type in data_filtered[rock_type_column].unique():
            rock_data = data_filtered[data_filtered[rock_type_column] == rock_type]
            rock_data_transformed = np.log1p(rock_data[oxide_columns])
            cov_matrix = rock_data_transformed.cov()
            mean_vector = rock_data_transformed.mean()
            
            fake_transformed = np.random.multivariate_normal(mean_vector, cov_matrix, size=n_samples)
            fake_original_scale = np.expm1(fake_transformed)
            fake_rock_data = pd.DataFrame(fake_original_scale, columns=oxide_columns)
            fake_rock_data[rock_type_column] = rock_type
            fake_data.append(fake_rock_data)

        fake_data_combined = pd.concat(fake_data, ignore_index=True)

        # Calculate correlation matrices
        real_corr = data_filtered[oxide_columns].corr()
        fake_corr = fake_data_combined[oxide_columns].corr()
        diff_corr = np.abs(real_corr - fake_corr)

        # Display correlation matrices
        st.header("Correlation Matrices")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.heatmap(real_corr, ax=axes[0], cmap="coolwarm", annot=True, fmt=".2f")
        axes[0].set_title("Real Data Correlation Matrix")
        
        sns.heatmap(fake_corr, ax=axes[1], cmap="coolwarm", annot=True, fmt=".2f")
        axes[1].set_title("Fake Data Correlation Matrix")
        
        st.pyplot(fig)
        
        # Display difference matrix
        st.subheader("Difference in Correlation Matrices (Real vs Fake)")
        fig_diff, ax_diff = plt.subplots(figsize=(8, 6))
        sns.heatmap(diff_corr, cmap="viridis", annot=True, fmt=".2f", ax=ax_diff)
        ax_diff.set_title("Difference Matrix")
        st.pyplot(fig_diff)

        # Show mean absolute difference
        mean_diff = diff_corr.mean().mean()
        st.write(f"**Mean Absolute Difference in Correlations:** {mean_diff:.4f}")

        # Provide a download button for fake data
        st.header("Download Fake Data")
        csv = fake_data_combined.to_csv(index=False)
        st.download_button(
            label="Download Fake Data as CSV",
            data=csv,
            file_name="fake_data.csv",
            mime="text/csv"
        )
