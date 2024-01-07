#!/usr/bin/env python
# coding: utf-8

# # project 2 = car price prediction

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import seaborn as sns


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


from sklearn.linear_model import LinearRegression


# In[7]:


from sklearn import metrics


# In[8]:


car_dataset = pd.read_csv('car_price_data.csv')


# In[9]:


car_dataset.head()


# In[10]:


car_dataset.shape


# In[11]:


car_dataset.info()


# In[12]:


car_dataset.isnull().sum()


# In[13]:


print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())


# In[14]:


car_dataset.replace({'Seller_Type':{'Dealer':0, 'Individual':1,}}, inplace = True)


# In[15]:


car_dataset.replace({'Fuel_Type':{'Petrol':0, 'Diesel':1, 'CNG':2}}, inplace = True)


# In[16]:


car_dataset.replace({'Transmission':{'Manual':0, 'Automatic':1}}, inplace = True)


# In[17]:


car_dataset.head()


# In[18]:


car_dataset.tail()


# In[19]:


X = car_dataset.drop(['Car_Name', 'Selling_Price'],axis = 1)


# In[20]:


Y = car_dataset['Selling_Price']


# In[21]:


print(X)


# In[22]:


print(Y)


# In[23]:


X_train, X_test, Y_train , Y_test = train_test_split(X,Y, test_size = 0.1, random_state = 2)


# In[24]:


lin_reg_model = LinearRegression()


# In[25]:


lin_reg_model.fit(X_train,Y_train)


# In[26]:


training_data_prediction = lin_reg_model.predict(X_train)


# In[27]:


error_score = metrics.r2_score(Y_train , training_data_prediction)


# In[28]:


print ("R squared error:", error_score)


# In[29]:


plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("predicted price")
plt.title("actual price vs predicted price")
plt.show()


# In[30]:


test_data_prediction = lin_reg_model.predict(X_test)


# In[31]:


error_score = metrics.r2_score(Y_test , test_data_prediction)
                               
                            


# In[32]:


print ("R squared error:", error_score)


# In[33]:


plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("predicted price")
plt.title("actual price vs predicted price")
plt.show()


# In[ ]:




