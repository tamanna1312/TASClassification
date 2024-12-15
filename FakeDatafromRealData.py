#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Loading our normalised real world data.
data = pd.read_csv('/Users/tamanna/Downloads/NormalisedFilteredGEOROCDatawithoutLabel.csv')
data


# In[3]:


#Counting the number of samples for each rock type.
rock_type_column = 'Rock_name'  
sample_counts = data[rock_type_column].value_counts()
sample_counts


# In[4]:


#Defining the threshold for the rock types with less number of samples.
threshold = 10000
fake_data = [] #Creating an empty list.

#Looping through each rock type and create fake data for underrepresented types.
for rock_type, count in sample_counts.items():
    if count < threshold:
        rock_data = data[data[rock_type_column] == rock_type]
        oxide_columns = rock_data.columns[1:]  
        rock_data_transformed = np.log1p(rock_data[oxide_columns]) #Log-transforming to reduce skewness and normalize distributions).
        cov_matrix = rock_data_transformed.cov() #Calculating covariance matrix  for correlation structure.
        mean_vector = rock_data_transformed.mean() #Calculateing mean vector for correlation structure.
        n_new_samples = threshold - count #Subtracting the already exisiting samples from the threshold value.
        #Creating fake data using multivariate normal distribution.
        fake_transformed = np.random.multivariate_normal(mean_vector, cov_matrix, size=n_new_samples)
        #Reversing log-transform to return to original scale.
        fake_original_scale = np.expm1(fake_transformed)
        fake_rock_data = pd.DataFrame(fake_original_scale, columns=oxide_columns)
        fake_rock_data[rock_type_column] = rock_type 
        fake_data.append(fake_rock_data)


# In[5]:


#Combining all fake data
fake_data = pd.concat(fake_data, ignore_index=True)
fake_data


# # Density plots to visualise the distribution of both real and fake data.

# In[6]:


fake_only =fake_data.copy()
real_data = data.copy()
oxide_columns = [col for col in real_data.columns if col != rock_type_column]
for oxide in oxide_columns:
    plt.figure(figsize=(8, 5))
    sns.kdeplot(real_data[oxide], label='Real Data', shade=True, color='blue')
    sns.kdeplot(fake_only[oxide], label='Fake Data', shade=True, color='orange')
    plt.xlabel(f"{oxide} (wt%)")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


# # Correlation Matrices of both real and fake data

# In[7]:


oxide_columns = [col for col in data.columns if col != rock_type_column]
#Correlation matrices.
real_corr = real_data[oxide_columns].corr()
fake_corr = fake_only[oxide_columns].corr()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(real_corr, ax=axes[0], cmap="coolwarm", annot=True, fmt=".2f")
axes[0].set_title("Real Data Correlation Matrix")

sns.heatmap(fake_corr, ax=axes[1], cmap="coolwarm", annot=True, fmt=".2f")
axes[1].set_title("Fake Data Correlation Matrix")

plt.show()

diff_corr = np.abs(real_corr - fake_corr)  
plt.figure(figsize=(8, 6))
sns.heatmap(diff_corr, cmap="viridis", annot=True, fmt=".2f")
plt.title("Difference in Correlation Matrices (Real vs Fake)")
plt.show()

mean_diff = diff_corr.mean().mean()
print(f"Mean Absolute Difference in Correlations: {mean_diff:.4f}")


# In[ ]:




