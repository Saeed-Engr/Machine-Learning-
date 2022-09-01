#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


dt = pd.read_csv("E:/MS_AI_IUB_DATASETS_2022/machine learning dataset/Ecommerce Customers")
dt.head()


# In[6]:


dt.shape


# In[7]:


dt.ndim


# In[8]:


dt.tail()


# In[10]:


dt.info()


# In[11]:


dt.describe()


# In[12]:


dt.isnull().sum()


# In[13]:


dt.isnull().any()


# In[65]:


dt.duplicated().sum()


# In[14]:


dt.columns


# In[ ]:





# In[16]:


# Correlation matrix
corr = dt.corr() 
plt.figure(figsize=(12,10))
sns.heatmap(data=corr, annot=True, cmap='Spectral').set(title="Correlation Matrix")


# In[19]:


dt.sample(3)


# In[24]:



#x=dt.drop(columns=['Yearly Amount Spent','Email','Address','Avatar']


# In[26]:


y=dt['Yearly Amount Spent']
print(y.shape)
print()
y.head()


# In[28]:


x=dt.drop(columns=['Yearly Amount Spent','Email','Address','Avatar'])


# In[29]:


x.head()


# In[30]:


print(x.shape)
print(y.shape)

# Spliting the data in test and train
# In[31]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,  train_size=0.7, random_state=1)


# In[34]:


# let's see the shape of each dataset

print(x.shape)
print(y.shape)
print()
print(x_train.shape)
print(y_train.shape)
print()
print(x_test.shape)
print(y_test.shape)


# In[35]:


from sklearn.linear_model import LinearRegression


# In[36]:


model = LinearRegression()


# In[37]:


model.fit(x_train,y_train)


# In[38]:


predict = model.predict(x_test)


# In[44]:


predict[:4]


# In[45]:


print("Accuracy of training dataset:", model.score(x_train,y_train))
print("Accuracy of test dataset:", model.score(x_test,y_test))


# # Actual VS Predicted

# In[68]:


y_pred_test = model.predict(x_test)
y_pred_train = model.predict(x_train)


# In[69]:


test = pd.DataFrame({
    'Y test':y_test,
    'Y test predicted':y_pred_test
})

train = pd.DataFrame({
    'Y train':y_train,
    'Y train predicted':y_pred_train
})


# In[70]:


test.sample(10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[46]:


import matplotlib.pyplot as plt 
plt.scatter(y_test,predict)
plt.xlabel('Y Test')
plt.ylabel('Y pridected')
plt.show()


# In[47]:


from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[48]:


from math import sqrt

print('MAE = ', mean_absolute_error(y_test,predict))


# In[49]:


print('MSE =', mean_squared_error(y_test,predict))


# In[50]:


print('RMSE = ', sqrt(mean_squared_error(y_test,predict)))


# In[51]:


dt.corr()


# In[52]:


dt.head(3)


# # prediction

# In[62]:


user = pd.DataFrame([[31.926272,11.109461,37.268959,2.664034]]) # [34.4972,12.6556,39.5776,4.0826]


# In[63]:


result = model.predict(user)
result


# In[ ]:




