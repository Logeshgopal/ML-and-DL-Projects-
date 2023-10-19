#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Prediction Project Using Logistic Regression
# 

# In[1]:


#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay,confusion_matrix


# In[3]:


#Import the Dataset
heart=pd.read_csv("heart_disease_data.csv")


# In[4]:


#First  rows of the dataset
heart.head()


# In[5]:


#Last  Rows of the dataset
heart.tail()


# In[6]:


# Number of Rows and Columns in the Dataset
heart.shape


# In[7]:


# Getting some info about data
heart.info()


# In[8]:


#Checking for missing values
heart.isnull().sum()


# In[9]:


#Statistical measures about the data
heart.describe()


# In[12]:


#Checking the distribution of target variable
counts=heart['target'].value_counts()
counts


# In[30]:


#Visulaization
plt.figure(figsize=(6,4))
plt.bar(counts.index,counts.values,color="g")
plt.show()


# In[80]:


sns.pairplot(heart)


# # 1--> Defective Heart
# 
# # 0--> Healthy Heart

# In[31]:


#Spliting the Features and Target
x=heart.drop(columns="target",axis=1)
y=heart['target']


# In[32]:


print(x)


# In[33]:


print(y)


# # Spliting the Data into Training Data & Testing Data 

# In[34]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)


# In[37]:


print(x.shape,y.shape)


# In[35]:


#Shape of training and testing data
#Data splitted 80% training data and 20% testing data
print(x_train.shape,x_test.shape)


# In[36]:


#Shape of training and testing data
print(y_train.shape,y_test.shape)


# # Model Training

# # Logistic Regression

# In[38]:


#Logistic Regression model
model=LogisticRegression()


# In[39]:


#Training the logisticRegression model wwith Training Data
model.fit(x_train,y_train)


# # Model Evaluation

# # Accuracy Score
# 

# In[40]:


# Accuracy  on training data
x_train_pred=model.predict(x_train)
training_data_accuracy= accuracy_score(x_train_pred,y_train)


# In[41]:


print("Accuracy on Training Data : ",training_data_accuracy)


# In[42]:


#Accuracy on Test data
x_test_pred=model.predict(x_test)
testing_data_accuracy= accuracy_score(x_test_pred,y_test)


# In[43]:


print("Accuracy on Testing Data : ",testing_data_accuracy)


# In[61]:


model.classes_


# In[79]:


y_prob=model.predict_proba(x_test)

y_pred_dict={"class 1":0,"class 0":0}

for y in y_prob:
    if y[0]>=0.5:
        y_pred_dict["class 0"] +=1
    else:
        y_pred_dict["class 1"] +=1
        
print(y_pred_dict)


# In[70]:


cnt=0

for i in x_test_pred:
    if i ==0:
        cnt+=1
print(cnt)


# # Building a predictive system

# In[55]:


input_data=(69,1,2,140,254,0,0,146,0,2,1,3,3)

#Change the input data to a numpy array
input_data_as_numpy=np.asarray(input_data)

# Reshape the numpy array as we are predicting for only on instance
input_data_reshaped=input_data_as_numpy.reshape(1,-1)

prediction=model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
    print("The person does not have a Heart Disease")
else:
    print("The person has Heart Disease")


# In[ ]:





# In[113]:


#Confusion Matrix
confusion_matrix(x_test_pred,y_test)


# In[57]:


#Accuracy
(23+27)/(23+27+6+5)


# In[76]:


from sklearn.metrics import ConfusionMatrixDisplay


# In[88]:


ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,x_test_pred),display_labels=model.classes_).plot()


# In[ ]:





# In[87]:


from sklearn.metrics import classification_report
print(classification_report(x_test_pred,y_test))


# In[109]:


tpm=tnm=fpm=fnm=0

y_test=pd.DataFrame(y_test)


# In[110]:


for(yt, yp) in zip(y_test.values, x_test_pred):
    if yt [0] == 0 and yp == 0:
        tnm += 1
    elif yt[0]==0 and yp ==1:
        fpm += 1
    elif yt[0]==1 and yp ==0:
        fnm += 1
    elif yt[0]==1 and yp==1:
        tpm += 1
        
print(tpm,tnm,fpm,fnm)


# In[115]:


from sklearn.metrics import roc_curve,roc_auc_score


# In[116]:


fpr,tpr,threshold=roc_auc_score(y_test,x_test_pred)

plt.plot(fpr,tpr)
plt.xlabel("True Positive Rate")
plt.ylabel("False Positive Rate")
plt.show()


# In[117]:


# Calculate the predicted probabilities for the positive class
y_test_pred_proba = model.predict_proba(x_test)[:, 1]

# Compute ROC curve points
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)

# Plot the ROC curve
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve")
plt.show()


# In[ ]:




