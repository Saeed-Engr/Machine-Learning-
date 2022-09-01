#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load libraries
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Load data with only two features
boston = load_boston()
features = boston.data[:,0:2]
target = boston.target

# Create linear regression
regression = LinearRegression()

# Fit the linear regression
model = regression.fit(features, target)


# In[ ]:





# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[17]:


df = pd.read_csv("E:/MS_AI_IUB_DATASETS_2022/machine learning dataset/boston.csv",  sep="\s+", skiprows=22, header=None)
df.head()                                      #  skip 1st 22 rows from data & header none mean no heading in data


# In[ ]:


import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]


# In[ ]:





# In[18]:


data = pd.read_csv("E:/MS_AI_IUB_DATASETS_2022/machine learning dataset/boston.csv")
data


# In[19]:


data.shape


# In[20]:


data.columns


# In[21]:


data.tail()


# In[22]:


data.isnull().any()


# In[23]:


data.describe()


# In[24]:


data.info()


# In[44]:


# summarize the data type and null values
# just for better visual
data_type = pd.DataFrame(data.dtypes).T.rename({0:'Column Data Type'})
null_value = pd.DataFrame(data.isnull().sum()).T.rename({0:'Null Values'})

# combine the data
data_info = data_type.append(null_value)
data_info


# In[45]:


data.isnull().any()


# In[26]:


data.isnull().sum()


# In[28]:


data["MEDV"]


# In[29]:


data.nunique()


# In[ ]:





# # Data Visualization

# In[30]:


data.head()


# In[ ]:





# In[43]:


# Correlation matrix
corr = data.corr() 
plt.figure(figsize=(12,10))
sns.heatmap(data=corr, annot=True, cmap='Spectral').set(title="Correlation Matrix")


# In[ ]:





# Correlation
# -
# - It's used to see the relationship between features
# - Correlation value is between -1 to 1
# - Correlation value = -1 means negative correlation. If 'X' goes bigger, 'Y' goes smaller
# - Correlation value = 1 means positive correlation. If 'X' goes bigger, 'Y' also goes bigger
# - Correlation value = 0 means, there's no correlation between 'X' and 'Y'

# In[47]:


corr_matrix = data.corr().round(2)
corr_matrix
  #               


# In[48]:


mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix, center=0, vmin=-1, vmax=1, mask=mask, annot=True, cmap='BrBG')
plt.show()


# In[ ]:





# - we can see that RM has a positive correlation with MEDV, just like the correlation heatmap above.

# - From a mathematic point of view, univariate linear regression is a way to find parameters for the linear equation:
# 
# - Y=a∗X+b 
# - Where,
# 
# Y  = output
# X  = input
# a  = slope of the line (gradient) aka regression coefficient
# b  = intercept
# We already have  X  and  Y  values, the linear regression algorithm will find the best  a  and  b  parameters for us.
# 
# Feature Selection
# -
# Since RM has the highest correlation to MEDV, I will use this attribute to make univariate linear regression.

# In[50]:


# I use [[]] to create a dataframe
# if you use [], it will create a series

X = data[['RM']]
X.head()


# In[51]:


Y = data[['MEDV']]
Y.head()


# Instantiating the Model
# -
# For this project, I will use the Scikit-learn library to make linear regression. But there are some notes when using this library:
# 
# Every model inside scikit-learn is saved as a class, not an instance
# We need to create an instance using the class of the model we want to use
# We can say that an instance of a class is an object of a class

# In[52]:


# import linear regression library
from sklearn.linear_model import LinearRegression

# instantiating the model
model = LinearRegression()


# Train - Test Split
# -
# We need to split our dataset into 2 datasets:
# 
# Train dataset, used to train our model. The machine will try to capture the pattern of our dataset. And that's what we called a model.
# Test dataset, used to test our model
# The rule of thumb for splitting datasets is 70% for train dataset and 30% for the test dataset.

# In[53]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=1)


# In[54]:


# let's see the shape of each dataset

print(X.shape)
print(Y.shape)
print()
print(X_train.shape)
print(Y_train.shape)
print()
print(X_test.shape)
print(Y_test.shape)


# In[55]:


model.fit(X_train, Y_train)


# Parameter Estimates
# -
# After training the model, sklearn's parameters are saved inside the model's object
# The name of that parameter is always followed by underscore symbol (_)

# In[56]:


# first, let's see the coefficient value (a)
model_coef = model.coef_
model_coef.round(2)


# In[57]:


# model intercept (b)
model_intercept = model.intercept_
model_intercept.round(2)


# Prediction
# -
# - From these parameters, we can form our equation:
# 
# - Y=a∗X+b 
# - Y=8.46∗X+(−30.57) 
# - Hence,
# 
# - MEDV=8.46∗RM−30.57 
# - Let's try to predict MEDV with new RM data. Make sure our input is a 2D array. We can use reshape(-1,1) to transform our data into 2D.

# In[58]:


# using built-in predict
new_RM = np.array([6.5]).reshape(-1,1)
model.predict(new_RM).round(2)


# In[59]:


# using equation
equation_predict = (model_coef * new_RM) + model_intercept
equation_predict.round(2)


# In[60]:


# predict test dataset
y_test_pred = model.predict(X_test)

# let's check the prediction and the actual value
print(Y_test[:5].values)
print()
print(y_test_pred[:5].round(2))


# Model Evaluation
# -
# Used to evaluate the performance of our model

# In[61]:


plt.scatter(X_test, Y_test, label='test data', color='k')
plt.plot(X_test, y_test_pred, label='pred data', color='b', linewidth=3)
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.title('Model Evaluation')
plt.legend(loc='upper left')

# plt.savefig('./output/linear-regression.png')
plt.show()


# In[62]:


# using function from scikit-learn
from sklearn.metrics import mean_squared_error

mean_squared_error(Y_test, y_test_pred).round(2)


# In[63]:


# step 1, calculate the difference
diff = (Y_test - Y_test.mean())
diff.head()


# In[ ]:




