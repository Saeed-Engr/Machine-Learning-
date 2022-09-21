#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Breast Cancer Survival Prediction with Machine Learning


# In[86]:


import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn .linear_model import LinearRegression
#Plotly Express provides functions to visualize a variety of types of data
import plotly.express as px
from sklearn.model_selection import train_test_split
#SVC. It is C-support vector classification whose implementation is based on libsvm. The module used by scikit-learn is sklearn. svm. SVC
from sklearn.svm import SVC
# read data 
data =pd.read_csv("C://Users//usman shafeeq//Desktop//DataSet//BRCA.csv")
# Print data head
print(data.head())


# In[87]:


# check the columns of this dataset contains any null values or not:
print(data.isnull().sum())


# In[88]:


# this dataset has some null value in each column, I will drop these null values.
data = data.dropna()


# In[89]:


# look at the insights about the columns of this data.
data.info()


# In[90]:


# Look the Gender column to see how many females and males are there.
print(data.Gender.value_counts())


# In[91]:


# look at the type of surgeries done to the patients.
# surgery _type 
surgery =data["Surgery_type"].value_counts()
transactions = surgery.index
quantity = surgery.values
figure = px.pie(data,values=quantity,names=transactions,hole=0.5,title='Type of Surgery of Patients')
figure.show()


# In[92]:


# transform the values of all the categorical columns.
data["Tumour_Stage"] = data["Tumour_Stage"].map({"I": 1, "II":2, "III": 3})
data["Histology"] = data["Histology"].map({"Infiltrating Ductal Carcinoma": 1,"Infiltrating Lobular Carcinoma":2,"Mucinous Carcinoma": 3})
data["ER status"] = data["ER status"].map({"Positive": 1})
data["PR status"]= data["PR status"].map({"Positive": 1})
data["HER2 status"]=data["HER2 status"].map({"Positive" :1,"Negative":2})
data["Gender"]= data["Gender"].map({"MALE":0,"FEMALE":1})
data["Surgery_type"] = data["Surgery_type"].map({"Other" :1, "Modified Radical Mastectomy":2,"Lumpectomy": 3,"Simple Mastectomy" :4})
data["Patient_Status"] = data["Patient_Status"].map({"Alive" :0,"Dead" :1})

print (data.head())


# In[93]:


# split the data into training and test set
# splitting data

x = np.array(data[['Age', 'Gender', 'Protein1', 'Protein2', 'Protein3','Protein4', 
                   'Tumour_Stage', 'Histology', 'ER status', 'PR status', 
                   'HER2 status', 'Surgery_type']])
y = np.array(data[['Patient_Status']])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)


# In[94]:



Xtrain,xtest,Ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=42)


# In[95]:


model =LinearRegression()
model.fit(Xtrain,Ytrain)
print(model.score(xtest,ytest))


# In[96]:


# Prediction
# features = [['Age', 'Gender', 'Protein1', 'Protein2', 'Protein3','Protein4', 'Tumour_Stage', 'Histology', 'ER status', 'PR status', 'HER2 status', 'Surgery_type']]
features = np.array([[36.0, 1, 0.080353, 0.42638, 0.54715, 0.273680, 3, 1, 1, 1, 2, 2,]])
print(model.predict(features))


# In[ ]:





# In[ ]:




