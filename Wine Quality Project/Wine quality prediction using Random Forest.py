#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Prediction

# In[2]:


#Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[3]:


#Importing the dataset
wine=pd.read_csv("winequality-red.csv")


# In[4]:


#Top 5 rows
wine.head()


# In[5]:


#Last 5 rows
wine.tail()


# In[6]:


#Shape of the dataset
wine.shape


# In[7]:


#Checking for missing values
wine.isnull().sum()


# In[8]:


#info
wine.info()


# # Data Analysis and Visulaization

# In[9]:


#Statistical Information on dataset
wine.describe()


# In[10]:


#check the value counts of quality
wine['quality'].value_counts()


# In[15]:


#Number of values for each quality
sns.countplot(data=wine,x='quality')


# In[16]:


#Volatile acidity vs Quality

plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='volatile acidity',data=wine)


# In[17]:


# Citric acid vs quality

plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='citric acid',data=wine)


# In[18]:


#residual sugar
plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='residual sugar',data=wine)


# # Correlation

# In[19]:


# Types of Correlation

#1.Postive Correlation
#2.Negative Corelation

correlation=wine.corr()


# In[24]:


#Constructing a heat map to understand the correlation between columns
plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True,square=True,fmt='.2f',annot=True,annot_kws={'size':8},cmap='Blues')


# # Data Preprocessing

# In[25]:


#Separate the data and label
x=wine.drop('quality',axis=1)


# In[26]:


print(x)


# In[27]:


#Label Binarization
y=wine['quality'].apply(lambda y_value:1 if y_value >=7 else 0)


# In[28]:


print(y)


# In[30]:


y.value_counts()


# # Train and Test split

# In[31]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[32]:


print(x.shape,x_train.shape,x_test.shape)


# In[33]:


print(y.shape,y_train.shape,y_test.shape)


# # ModelTraining:
# 
# # Random Forest Classifier

# In[35]:


model=RandomForestClassifier()


# In[36]:


model.fit(x_train,y_train)


# # Model evaluation

# In[37]:


#Accuracy on test data

x_test_pred=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_pred,y_test)


# In[38]:


print("Accuracy on test data :",test_data_accuracy)


# # Building a predictive system

# In[43]:


input_data=(8.5,0.28,0.56,1.8,0.092,35.0,103.0,0.9969,3.3,0.75,10.5)

#Change the input data to a numpy array
input_data_numpy=np.asarray(input_data)

# Reshape the numpy array as we are predicting for only on instance
input_data_numpy_shaped=input_data_numpy.reshape(1,-1)

prediction=model.predict(input_data_numpy_shaped)
print(prediction)

if prediction[0]==0:
    print('Bad Quality Wine')
else:
    print("Good Quality Wine")


# In[ ]:




