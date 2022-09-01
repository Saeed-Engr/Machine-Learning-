#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[57]:


data = pd.read_csv("E:/MS_AI_IUB_DATASETS_2022/machine learning dataset/boston.csv")
data.head()


# In[58]:


data.describe()


# In[59]:


data


# In[60]:


data.columns


# In[61]:


x = data.iloc[:,:13].values
y = data["MEDV"].values

print(x.shape)
print(y.shape)


# In[62]:


x


# In[63]:


y


# In[ ]:





# In[64]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test = train_test_split(x,y,test_size = 0.3 , random_state = 0 )


# In[65]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# In[66]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[67]:


model.fit(x_train,y_train)


# In[68]:


predict = model.predict(x_test)


# In[69]:


predict[:4]


# In[70]:


print("Accuracy of training dataset:", model.score(x_train,y_train))
print("Accuracy of test dataset:", model.score(x_test,y_test))


# In[ ]:





# In[71]:


13*13


# In[77]:


from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Flatten())
network.add(layers.Dense(64, activation='relu', input_shape=(169,)))
network.add(layers.Dense(32, activation='relu'))
network.add(layers.Dense(8, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))


# In[78]:


network.compile(optimizer = 'adam' , loss = 'binary_crossentropy',metrics = ['accuracy'] )


# In[79]:


network.fit(x_train , y_train , batch_size = 20 ,epochs = 10 )


# In[ ]:




