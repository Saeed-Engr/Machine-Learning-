#!/usr/bin/env python
# coding: utf-8

# Fitting a Line
# -
# Problem
# -
# - You want to train a model that represents a linear relationship between the feature and target vector.
# 
# Solution
# -
# - Use a linear regression (in scikit-learn, LinearRegression):

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


# In[2]:


# View the intercept
model.intercept_


# In[3]:


# View the feature coefficients
model.coef_


# In[4]:


# First value in the target vector multiplied by 1000
target[0]*1000


# In[5]:


# Predict the target value of the first observation, multiplied by 1000
model.predict(features)[0]*1000


# In[6]:


# First coefficient multiplied by 1000
model.coef_[0]*1000


# In[ ]:





# # Exploring Dataset 

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[17]:


from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Load data with only two features
boston = load_boston()


# In[18]:


features = boston.data
target = boston.target


# In[21]:


ta = pd.DataFrame(target)
ta.head()


# In[20]:


fe = pd.DataFrame(features)
fe.head()


# In[ ]:





# # Building Model with own model

# In[26]:


data = pd.read_csv("E:/MS_AI_IUB_DATASETS_2022/machine learning dataset/boston.csv")
data.head()


# In[ ]:





# In[ ]:





# In[27]:


# I use [[]] to create a dataframe
# if you use [], it will create a series

X = data[['RM']]
X.head()


# In[28]:


Y = data[['MEDV']]
Y.head()


# In[29]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=1)


# In[30]:


# let's see the shape of each dataset

print(X.shape)
print(Y.shape)
print()
print(X_train.shape)
print(Y_train.shape)
print()
print(X_test.shape)
print(Y_test.shape)


# In[31]:


model.fit(X_train, Y_train)


# In[32]:


# Load libraries
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Create linear regression
regression = LinearRegression()

# Fit the linear regression
model = regression.fit(X_train, Y_train)


# In[33]:


# View the intercept
model.intercept_


# In[34]:


# View the feature coefficients
model.coef_

# First value in the target vector multiplied by 1000
Y_train[0]  #*1000
# In[42]:


# Predict the target value of the first observation, multiplied by 1000
model.predict(X_train)[0]*1000


# In[43]:


# First coefficient multiplied by 1000
model.coef_[0]*1000


# In[ ]:





# # Handling Interactive Effects
Problem
You have a feature whose effect on the target variable depends on another feature.

Solution
Create an interaction term to capture that dependence using scikit-learn’s Polynomial
Features:
# In[1]:


# Load libraries
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures


# Load data with only two features
boston = load_boston()
features = boston.data[:,0:2]
target = boston.target


# In[2]:


# Create interaction term
interaction = PolynomialFeatures(
 degree=3, include_bias=False, interaction_only=True)
features_interaction = interaction.fit_transform(features)


# In[3]:


# Create linear regression
regression = LinearRegression()

# Fit the linear regression
model = regression.fit(features_interaction, target)


# In[4]:


# View the feature values for first observation
features[0]


# In[5]:


# Import library
import numpy as np

# For each observation, multiply the values of the first and second feature
interaction_term = np.multiply(features[:, 0], features[:, 1])


# In[7]:


interaction_term[0]


# In[8]:


# View the values of the first observation
features_interaction[0]


# In[ ]:





# # Fitting a Nonlinear Relationship

# In[9]:


# Load library
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures

# Load data with one feature
boston = load_boston()
features = boston.data[:,0:1]
target = boston.target

# Create polynomial features x^2 and x^3
polynomial = PolynomialFeatures(degree=3, include_bias=False)
features_polynomial = polynomial.fit_transform(features)

# Create linear regression
regression = LinearRegression()

# Fit the linear regression
model = regression.fit(features_polynomial, target)


# In[10]:


# View first observation
features[0]


# In[11]:


# View first observation raised to the second power, x^2
features[0]**2


# In[12]:


# View first observation raised to the third power, x^3
features[0]**3


# In[13]:


# View the first observation's values for x, x^2, and x^3
features_polynomial[0]


# In[ ]:





# # Reducing Variance with Regularization

# In[14]:


#Use a learning algorithm that includes a shrinkage penalty (also called regularization)
#like ridge regression and lasso regression:
# Load libraries

from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# Load data
boston = load_boston()
features = boston.data
target = boston.target

# Standardize features
scaler = StandardScaler()


# In[15]:


features_standardized = scaler.fit_transform(features)

# Create ridge regression with an alpha value
regression = Ridge(alpha=0.5)

# Fit the linear regression
model = regression.fit(features_standardized, target)


# In[16]:


# Load library
from sklearn.linear_model import RidgeCV

# Create ridge regression with three alpha values
regr_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])

# Fit the linear regression
model_cv = regr_cv.fit(features_standardized, target)

# View coefficients
model_cv.coef_


# In[17]:


# View alpha
model_cv.alpha_


# In[ ]:





# # Reducing Features with Lasso Regression

# In[18]:


#Use a lasso regression:
# Load library

from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# Load data
boston = load_boston()


# In[19]:


features = boston.data
target = boston.target

# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Create lasso regression with alpha value
regression = Lasso(alpha=0.5)

# Fit the linear regression
model = regression.fit(features_standardized, target)


# In[20]:


# View coefficients
model.coef_


# In[21]:


# Create lasso regression with a high alpha
regression_a10 = Lasso(alpha=10)
model_a10 = regression_a10.fit(features_standardized, target)
model_a10.coef_


# In[ ]:




