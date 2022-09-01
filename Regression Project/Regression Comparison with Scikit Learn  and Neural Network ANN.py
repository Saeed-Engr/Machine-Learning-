#!/usr/bin/env python
# coding: utf-8

# In[49]:


import torch
torch. __version__


# In[ ]:





# In[79]:


from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
sns.set()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[ ]:





# In[80]:


#dataset = pd.read_csv("E:/MS_AI_IUB_DATASETS_2022/machine learning dataset/boston.csv")

# from sklearn.datasets import load_boston
dataset = load_boston()


# In[81]:


df=pd.DataFrame(dataset.data)
df.columns=dataset.feature_names
df["Price"]=dataset.target
df.head()


# In[ ]:





# In[82]:


TargetName = "PRICES"
FeaturesName = [
              #-- "Crime occurrence rate per unit population by town"
              "CRIM",
              #-- "Percentage of 25000-squared-feet-area house"
              'ZN',
              #-- "Percentage of non-retail land area by town"
              'INDUS',
              #-- "Index for Charlse river: 0 is near, 1 is far"
              'CHAS',
              #-- "Nitrogen compound concentration"
              'NOX',
              #-- "Average number of rooms per residence"
              'RM',
              #-- "Percentage of buildings built before 1940"
              'AGE',
              #-- 'Weighted distance from five employment centers'
              "DIS",
              ##-- "Index for easy access to highway"
              'RAD',
              ##-- "Tax rate per $100,000"
              'TAX',
              ##-- "Percentage of students and teachers in each town"
              'PTRATIO',
              ##-- "1000(Bk - 0.63)^2, where Bk is the percentage of Black people"
              'B',
              ##-- "Percentage of low-class population"
              'LSTAT',
              ]


# In[88]:


y=df["Price"]
x=df.drop("Price",axis=1)


# In[89]:


from sklearn.preprocessing import StandardScaler
sscaler = StandardScaler()
sscaler.fit(x)
X_std= sscaler.transform(x)


# In[ ]:





# In[91]:


#from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_std,y,test_size=0.3,random_state=99)
X_train.shape


# In[102]:


# let's see the shape of each dataset

print(X_train.shape)
print(y_train.shape)
print()
print(y_test.shape)
print(X_test.shape)


# In[124]:


# import linear regression library
from sklearn.linear_model import LinearRegression

# instantiating the model
model = LinearRegression()


# In[125]:


model.fit(X_train, y_train)


# In[126]:


predict = model.predict(X_test)


# In[127]:


predict[:4]


# In[128]:


print("Accuracy of training dataset:", model.score(X_train, y_train))
print("Accuracy of test dataset:", model.score(X_test,y_test))


# In[ ]:





# In[129]:


# first, let's see the coefficient value (a)
model_coef = model.coef_
model_coef.round(2)


# In[130]:


# model intercept (b)
model_intercept = model.intercept_
model_intercept.round(2)


# In[131]:


# predict test dataset
y_test_pred = model.predict(X_test)

# let's check the prediction and the actual value
print(y_test[:5].values)
print()
print(y_test_pred[:5].round(2))


# In[ ]:





# In[ ]:





# In[132]:


class NN(nn.Module):
  def __init__(self):
    super(NN,self).__init__()
    self.layer1=nn.Linear(X_train.shape[1],39)
    self.layer2=nn.Linear(39,26)
    self.layer3=nn.Linear(26,13)
    self.layer4=nn.Linear(13,1)
  def forward(self,x):
    x=F.relu(self.layer1(x))
    x=F.relu(self.layer2(x))
    x=F.relu(self.layer3(x))
    x=self.layer4(x)
    return x
model = NN()
print(model)


# In[133]:


x=torch.tensor(np.array(X_train),dtype=torch.float32,requires_grad=True)
y=torch.tensor(np.array(y_train).reshape(-1,1),dtype=torch.float32)
x


# In[134]:


#import torch.optim as optim
optimizer= optim.SGD(model.parameters(),lr=0.003)


# In[135]:


loss_fn=nn.MSELoss()


# In[136]:


epochs=200
for i in range(epochs):
  #initialize the model parameter
  optimizer.zero_grad(set_to_none=True)
  #calculate the loss
  output=model(x)
  loss=loss_fn(output,y)
  #backpropagation
  loss.backward()
  #update the parameters
  optimizer.step()
  if(i%5==0):
    print(f"epochs: {i}......loss:{loss}")


# In[137]:


y_train_pred = model(torch.tensor(X_train,dtype=torch.float32,requires_grad=True))
y_test_pred = model(torch.tensor(X_test,dtype=torch.float32))

#convert to numpy array
y_train_pred = y_train_pred.detach().numpy()
y_test_pred = y_test_pred.detach().numpy()


# In[138]:


test_accuracy=r2_score(y_test,y_test_pred)
train_accuracy=r2_score(y_train,y_train_pred)
print(train_accuracy)
print(test_accuracy)


# In[139]:


plt.xlabel("Price")
plt.ylabel("Predicted Price")
plt.scatter(y_train,y_train_pred,color='r',label="train_data")
plt.scatter(y_test,y_test_pred,color='b',label="test_data")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




