#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation using K-Means Clustering

# # About Dataset
# 
# - Content:
#     
# You are owing a supermarket mall and through membership cards ,
# you have some basic data about your customers like Customer ID, age, gender, annual income and spending score.
# Spending Score is something you assign to the customer based on your defined parameters like customer behavior and 
# purchasing data.
# 
# - Problem Statement:
#     
# You own the mall and want to understand the customers like who can be easily converge [Target Customers] 
# so that the sense can be given to marketing team and plan the strategy accordingly.

# ### Importing the Dependencies

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# ### Data Collection & Analysis

# In[2]:


# loading the data from csv file to a Pandas DataFrame
customer_data = pd.read_csv('Mall_Customers.csv')


# In[3]:


# first 5 rows in the dataframe
customer_data.head()


# In[4]:


# finding the number of rows and columns
customer_data.shape


# In[5]:


# getting some informations about the dataset
customer_data.info()


# In[6]:


# checking for missing values
customer_data.isnull().sum()


# ### Choosing the Annual Income Column & Spending Score column

# In[7]:


X = customer_data.iloc[:,[3,4]].values


# In[8]:


print(X)


# ### Choosing the number of clusters

# - WCSS -> Within Clusters Sum of Squares

# In[10]:


# finding wcss value for different number of clusters

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[15]:


pd.DataFrame(wcss)


# In[17]:


# plot an elbow graph

sns.set()
plt.plot(range(1,11), wcss)
#Title
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
#plot
plt.show()


# ### Optimum Number of Clusters = 5

# ### Training the k-Means Clustering Model

# In[12]:


kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

# return a label for each data point based on their cluster
Y = kmeans.fit_predict(X)

print(Y)


# 5 Clusters - 0, 1, 2, 3, 4

# ### Visualizing all the Clusters

# In[29]:


# plotting all the clusters and their Centroids

plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='blue', label='Cluster 5')

# plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='black',marker='', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()


# In[ ]:





# In[ ]:




