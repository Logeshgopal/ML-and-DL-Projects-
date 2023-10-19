#!/usr/bin/env python
# coding: utf-8

# # Fake News Prediction

# In[1]:


#Importing the dependencies

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


import nltk
nltk.download('stopwords')


# In[4]:


#Printing the stopwords
print(stopwords.words('english'))


# # Data Pre-processing

# In[5]:


#Loading the dataset
news=pd.read_csv("train.csv")


# In[6]:


#Top 5 rows
news.head()


# In[8]:


#Last 5 rows
news.tail()


# In[7]:


#Shape
news.shape


# In[9]:


#Missing value counting
news.isnull().sum()


# In[10]:


#Replacing the Null values with empty string
news=news.fillna('')


# In[11]:


#Merging the author name and news title
news['content']=news['author']+' '+news['title']


# In[13]:


print(news['content'])


# In[14]:


#Separate the data & Label
x=news.drop(columns='label',axis=1)
y=news['label']


# In[15]:


print(x)


# In[16]:


print(y)


# In[17]:


#Stemming:

#Stemming is the process of reducing a word to its root word
#example: actor,actress,acting ---> act


# In[18]:


port=PorterStemmer()


# In[19]:


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# In[20]:


news['content']=news['content'].apply(stemming)


# In[21]:


news['content']


# In[22]:


#Separate the data and label
x=news['content'].values
y=news['label'].values


# In[23]:


print(x)


# In[24]:


print(y)


# In[25]:


y.shape


# In[28]:


#converting the textual data to numerical data
vectorizer=TfidfVectorizer()
vectorizer.fit(x)

x=vectorizer.transform(x)


# In[29]:


print(x)


# In[30]:


#Spliting dataset to training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)


# # Training Model:Logistic Regression

# In[31]:


model=LogisticRegression()


# In[32]:


model.fit(x_train,y_train)


# # Evaluation

# In[34]:


#Accuracy score on the training data

x_train_pred=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_pred,y_train)


# In[35]:


print("Accuracy on training data :",training_data_accuracy)


# In[38]:


#Accuracy score on the testing data

y_pred=model.predict(x_test)
testing_data_accuracy=accuracy_score(y_pred,y_test)


# In[39]:


print("Accuracy on testing data :",testing_data_accuracy)


# # Making a predictive system

# In[44]:


x_new=x_test[1]

prediction = model.predict(x_new)
print(prediction)


if prediction[0]==0:
    print("The news is Real")
else:
    print("The news is Fake")


# In[45]:


print(y_test[1])


# In[ ]:




