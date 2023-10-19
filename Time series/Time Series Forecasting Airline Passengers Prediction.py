#!/usr/bin/env python
# coding: utf-8

# # TIME SERIES FORECASTING FOR AIRLINE PASSENGERS

# In[2]:


#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#Load Specific forecasting tools
from statsmodels.tsa.arima_model import ARMA,ARIMA,ARIMAResults,ARMAResults
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pmdarima import auto_arima

#ignore harmless error
import warnings
warnings.filterwarnings("ignore")


# In[11]:


#Load the dataset
airline=pd.read_csv("AirPassengers.csv",index_col=0,parse_dates=True)
airline.index.freq='MS'


# In[12]:


#First 5 rows
airline.head()


# In[13]:


#Last 5 rows
airline.tail()


# In[14]:


#Shape 
airline.shape


# In[18]:


#plot dataset
airline['#Passengers'].plot(figsize=(12,5)).autoscale(axis='X',tight=True)


# In[19]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[20]:


result=seasonal_decompose(airline['#Passengers'],model="Multiplicative")
result.plot();


# In[21]:


stepwise=auto_arima(airline['#Passengers'],start_p=1,start_q=1,
                   max_p=3,max_q=3,m=12,start_P=0,seasonal=True,
                   d=None,D=1,trace=True,error_action='ignore',
                   suppress_warnings=True,stepwise=True)
stepwise.summary()


# In[22]:


from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.stattools import adfuller


# In[23]:


airline['d1']=diff(airline['#Passengers'],k_diff=1)


# In[24]:


print("Augumented Dickey-Fuller Test on Airline Data")
df_test=adfuller(airline['d1'].dropna(),autolag='AIC')
df_test


# In[ ]:





# In[25]:


print("Augumented Dickey-Fuller Test on Airline Data")

dfout=pd.Series(df_test[0:4],index=['ADF test statistics','p-value','# lags used','#Observation'])

for key,val in df_test[4].items():
    dfout[f'Critical value ({key})']=val
print(dfout)


# In[26]:


from statsmodels.tsa.stattools import acf,pacf


# In[28]:


#acf plot
acf(airline['#Passengers'])


# In[30]:


pacf(airline['#Passengers'])


# In[31]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[32]:


title="Autocorelation: Airline Passengers"
lags=40
plot_acf(airline['#Passengers'],title=title,lags=lags);


# In[33]:


title="Partial Autocorelation: Airline Passengers"
lags=40
plot_acf(airline['#Passengers'],title=title,lags=lags);


# In[34]:


airline.shape


# In[35]:


#set one year for testing
train=airline.iloc[:132]
test=airline.iloc[132:]


# In[39]:


from statsmodels.tsa.arima.model import ARIMA


# In[40]:


model=ARIMA(train['#Passengers'],order=(0,1,1))


# In[41]:


result=model.fit()
result.summary()


# In[42]:


#Obtain predicted value
start=len(train)
end=len(train)+len(test)-1
prediction=result.predict(start=start,end=end,dynamic=False,typ='levels').rename('ARIMA(0,1,1)Prediction')


# In[44]:


#Compare prediction to expected value
for i in range(len(prediction)):
    print(f" Predicted={prediction[i]}, expected={test['#Passengers'][i]}")


# In[48]:


#plots prediction against known value
ax=test['#Passengers'].plot(legend=True,figsize=(12,6))
prediction.plot(legend=True)
ax.autoscale(axis='x',tight=True)


# In[50]:


from sklearn.metrics import mean_squared_error


# In[51]:


error=mean_squared_error(test['#Passengers'],prediction)
print(f"ARIMA(0,1,1) MSE Error: {error:11.10}")


# In[53]:


from statsmodels.tools.eval_measures import rmse
error=rmse(test['#Passengers'],prediction)
print(f" ARIMA(0,1,1) MSE Error:{error:11.10}")


# # Retain  the model on the full data, and forecast the future

# In[54]:


model=ARIMA(airline['#Passengers'],order=(0,1,1))
result=model.fit()


# In[58]:


fcast=result.predict(len(airline),len(airline)+12*10,typ='levels').rename('ARIMA(0,1,1) Forecast')


# In[59]:


#Plot prediction
ax=airline['#Passengers'].plot(legend=True,figsize=(12,6))
fcast.plot(legend=True)
ax.autoscale(axis='x',tight=True)


# In[60]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[62]:


model=SARIMAX(train['#Passengers'],order=(0,1,1),seasonal_order=(2,1,1,12),enforce_invertibility=False)
result=model.fit()
result.summary()


# In[63]:


#Obtain predicted value
start=len(train)
end=len(train)+len(test)-1
prediction=result.predict(start=start,end=end,dynamic=False,typ='levels').rename('SARIMA(0,1,1)Prediction')


# In[64]:


#Compare prediction to expected value
for i in range(len(prediction)):
    print(f" Predicted={prediction[i]}, expected={test['#Passengers'][i]}")


# In[65]:


#plots prediction against known value
ax=test['#Passengers'].plot(legend=True,figsize=(12,6))
prediction.plot(legend=True)
ax.autoscale(axis='x',tight=True)


# In[66]:


from sklearn.metrics import mean_squared_error


# In[72]:


error=mean_squared_error(test['#Passengers'],prediction)
print(f"SARIMAX(0,1,1) MSE Error: {error:11.10}")


# In[80]:


from statsmodels.tools.eval_measures import rmse
error=rmse(test['#Passengers'],prediction)
print(f" SARIMAX(0,1,1) RMSE Error:{error:11.10}")


# # Retrain the model on the full data,forecast the future

# In[84]:


model=SARIMAX(train['#Passengers'],order=(1,1,0),seasonal_order=(2,1,1,12),enforce_stationarity=False)
result=model.fit()
fcast=result.predict(len(airline),len(airline)+12*10,typ='levels').rename('SARIMAX(0,1,1)Forecast')


# In[85]:


#Plot prediction
ax=airline['#Passengers'].plot(legend=True,figsize=(12,6))
fcast.plot(legend=True)
ax.autoscale(axis='x',tight=True)


# In[ ]:




