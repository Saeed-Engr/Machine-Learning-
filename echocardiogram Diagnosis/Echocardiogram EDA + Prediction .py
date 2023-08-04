#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[135]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt #for data visualizing
import seaborn as sns 
color = sns.color_palette()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[136]:


"""import pandas as pd
da = pd.read_fwf("E:/Paid Projects Dataset and Codes Upwork/echocardiogram.data")
da.head()
da.shape
"""


# # Reading and Exploring Data

# In[137]:


data = pd.read_csv('E:/Paid Projects Dataset and Codes Upwork/echocardiogram.csv')
data.head()


# In[138]:


data.shape


# In[139]:


data.info()


# In[140]:


data.tail()


# In[141]:


data.shape


# In[ ]:





# # Checking Missing Data

# In[142]:


print('Total cols : ',data.shape[1],' and total rows : ',data.shape[0])


# In[143]:


data.isnull().any()


# In[144]:


print('Missing data sum :')
print(data.isnull().sum())

print('\nMissing data percentage (%):')
print(data.isnull().sum()/data.count()*100)


# In[145]:


# summarize the data type and null values
# just for better visual
data_type = pd.DataFrame(data.dtypes).T.rename({0:'Column Data Type'})
null_value = pd.DataFrame(data.isnull().sum()).T.rename({0:'Null Values'})

# combine the data
data_info = data_type.append(null_value)
data_info


# In[146]:


x=data.isnull().sum()
x=x.sort_values(ascending=False)

plt.figure(figsize=(16,6))
ax= sns.barplot(x.index, x.values, alpha=0.9,color=color[9])
locs, labels = plt.xticks()
plt.setp(labels, rotation=60)
plt.title("Missing value checking",fontsize=20)
plt.ylabel('Missing value sum', fontsize=16)
plt.xlabel('Col name', fontsize=16)
plt.show()


# In[ ]:





# # Data Visualization

# In[147]:


data.sample(5)


# In[ ]:





# Correlation
# -
# It's used to see the relationship between features
# 
# - Correlation value is between -1 to 1
# - Correlation value = -1 means negative correlation. If 'X' goes bigger, 'Y' goes smaller
# - Correlation value = 1 means positive correlation. If 'X' goes bigger, 'Y' also goes bigger
# - Correlation value = 0 means, there's no correlation between 'X' and 'Y'

# In[148]:


# Correlation matrix
corr = data.corr() 
plt.figure(figsize=(12,10))
sns.heatmap(data=corr, annot=True, cmap='Spectral').set(title="Correlation Matrix")


# In[149]:


corr_matrix = data.corr().round(2)
corr_matrix              


# In[150]:


mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix, center=0, vmin=-1, vmax=1, mask=mask, annot=True, cmap='BrBG')
plt.show()


# In[151]:


import seaborn as sns

corelation = data.corr()
sns.heatmap(corelation, xticklabels = corelation.columns , yticklabels = corelation.columns  )
plt.show()


# # Number of people who have disease vs age

# In[152]:


plt.figure(figsize=(15,6))
sns.countplot(x='age',data = data, hue = 'alive',palette='husl')
plt.show()


# In[153]:


data.plot("age")
plt.show()


# In[154]:


data.plot("alive")
plt.show()


# In[155]:


sns.distplot(data['age'])
plt.show()


# In[156]:


sns.distplot(data['alive'])
plt.show()


# In[157]:


x = data['age']
y = data['alive']
x1 =  (x,y)
plt.hist(x1)
plt.show() 


# In[158]:


import seaborn as sns

sns.factorplot('age', data = data, kind = 'count')
plt.show()


# In[159]:


import seaborn as sns

sns.factorplot('alive', data = data, kind = 'count')
plt.show()


# In[ ]:





# # Some observations about the data:

# Categorical Features
# -

# In[160]:


cat_features = [feature for feature in data.columns if data[feature].dtypes == 'O']
print('Number of categorical variables: ', len(cat_features))
print('*'*80)
print('Categorical variables column name:',cat_features)


# numerical_features

# In[161]:


numerical_features = [feature for feature in data.columns if data[feature].dtypes != 'O']
print('Number of numerical variables: ', len(numerical_features))
print('*'*80)
print('Numerical Variables Column: ',numerical_features)


# In[ ]:





# Visualizing Categorical Features
# -

# In[162]:


for col in cat_features[:-1]:
    plt.figure(figsize=(6,3), dpi=100)
    sns.countplot(data=data,x=col,hue ='alive',palette='gist_rainbow_r')
    plt.legend(loc=(1.05,0.5))


# In[163]:


def get_insights(data):
    for cols in cat_features[:-1]:
        cat_group = data.groupby(cols)['group'].describe()
        percent = 100 *(cat_group['freq']/cat_group['count'])
        print('*'*70)
        print("Insights from '{}' data".format(cols))
        print('*'*70)
        for i in range(0,(len(cat_group))):
            print('{} from {} are {}'.format(round(percent[i],2),percent.index[i], cat_group['top'][i]))
        print('*'*70)


# In[164]:


get_insights(data)


# In[165]:


#Plotting the barplot of numerical features
for col in numerical_features:
    plt.figure(figsize=(6,3), dpi=100)
    sns.barplot(data=data,x='alive',y=col,palette='gist_rainbow_r')


# In[ ]:





# # Drop Columns

# In[167]:


data = data.drop(['name', 'group', 'aliveat1'], axis=1)
data.head()


# In[168]:


features_with_null = [features for features in data.columns if data[features].isnull().sum()>0]
for feature in features_with_null:
    print(feature, ':', round(data[feature].isnull().mean(), 4), '%')


# In[ ]:





# In[169]:


for feature in features_with_null:
    print(feature, ':', data[feature].unique())


# In[170]:


data = data.dropna(subset=['alive'])
data['alive'].isnull().sum()


# In[171]:


discrete_features = ['pericardialeffusion']
continuous_features = data.drop(['pericardialeffusion', 'alive'], 1).columns
label = ['alive']

print(continuous_features)


# In[172]:


for feature in discrete_features:
    data[feature] = data[feature].fillna(data[feature].mode()[0])


# In[ ]:





# In[173]:


for feature in continuous_features:
    data.boxplot(feature)
    plt.title(feature)
    plt.show()


# In[174]:


features_with_outliers = ['wallmotion-score', 'wallmotion-index', 'mult']


# In[175]:


for feature in continuous_features:
    if feature in features_with_outliers:
        data[feature].fillna(data[feature].median(), inplace=True)
    else:
        data[feature].fillna(data[feature].mean(), inplace=True)


# In[176]:


from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor()
outliers_rows = lof.fit_predict(data)


# In[177]:


mask = outliers_rows != -1


# In[178]:


data.isnull().sum()


# In[ ]:





# # Feature Selection

# In[ ]:





# In[179]:


from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy


# In[180]:


#define the features
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


from sklearn import preprocessing
from sklearn import utils

#convert y values to categorical values

lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)

#view transformed values
#print(y_transformed)



forest = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight='balanced')

# define Boruta feature selection method
feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2)
 
# find all relevant features
feat_selector.fit(X, y_transformed)


# In[181]:


# show the most important features
most_important = data.columns[:-1][feat_selector.support_].tolist()
most_important


# In[183]:


# select the top 7 features
top_features = data.columns[:-1][feat_selector.ranking_ <=7].tolist()
top_features


# In[ ]:





# In[184]:


data.columns


# # Statistics on the top features

# In[185]:


import statsmodels.api as sm


# In[186]:


X_top = data[top_features]
y = data['alive']


# In[187]:


res = sm.Logit(y,X_top).fit()
res.summary()


# In[ ]:





# In[ ]:





# In[188]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[189]:


data = data[mask]


# In[190]:


from sklearn.preprocessing import StandardScaler

data1 = pd.get_dummies(data, columns = discrete_features, drop_first = True)
scaler = StandardScaler()
data1[continuous_features] = scaler.fit_transform(data1[continuous_features])


# In[191]:


data1.head()


# In[192]:


X = data1.drop(['alive'], 1)
y = data1['alive']


# In[ ]:





# # Splitting data into Training and Testing

# Splitting 70% training data and 30% testing data

# In[205]:


#Importing our ML toolkit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, plot_confusion_matrix,classification_report
from sklearn.svm import SVC
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


# In[206]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape, X_test.shape


# In[ ]:





# # Building Models

# In[ ]:





# In[207]:


accuracy = {}


# In[216]:


model1 = LogisticRegression(max_iter=200)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
print(accuracy_score(y_test, y_pred1))
accuracy[str(model1)] = accuracy_score(y_test, y_pred1)*100


# In[217]:


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred1)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# In[219]:


print(classification_report(y_test,y_pred1))


# In[277]:


# ROC curve and AUC 
probs = model1.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
log_auc = roc_auc_score(y_test, probs)

# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot curve
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.title(f"AUC = {round(log_auc,3)}")
plt.show()


# Save the model of LogisticRegression
# -

# In[ ]:





# In[220]:


import pickle

# save the model to disk
filename = 'LogisticRegression.sav'
pickle.dump(model1, open(filename, 'wb'))
 


# In[221]:



# load the model from disk

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score( X_test,  y_test)


# In[225]:


result


# # Actual VS Predicted

# In[231]:


y_pred_test = loaded_model.predict(X_test)
y_pred_train = loaded_model.predict(X_train)


# In[239]:


test = pd.DataFrame({
    'Actual':y_test,
    'Y test predicted':y_pred_test
})

train = pd.DataFrame({
    'Actual':y_train,
    'Y train predicted':y_pred_train
})


# In[240]:


test.sample(10)


# In[245]:


X_test.head()


# In[269]:


data = X_test[:1]

data


# In[270]:


prediction = loaded_model.predict(data)

print("You are not at risk") if prediction[0] == 0 else print("You are at risk")


# In[ ]:





# In[271]:


# predict test dataset
y_test_pred = loaded_model.predict(X_test)

# let's check the prediction and the actual value
print(y_test[:15].values)
print()
print(y_test_pred[:15].round(2))


# In[ ]:





# # DecisionTreeClassifier

# In[ ]:





# In[272]:


model2 = DecisionTreeClassifier(max_depth=3)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print(accuracy_score(y_test, y_pred2))
accuracy[str(model2)] = accuracy_score(y_test, y_pred2)*100


# In[273]:


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred2)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# In[280]:


print(classification_report(y_test,y_pred2))


# In[295]:


from sklearn.metrics import f1_score
from sklearn.metrics import recall_score,precision_score,classification_report,roc_auc_score,roc_curve

logistic_f1 = f1_score(y_test, y_pred2)
print(f'The f1 score for DecisionTreeClassifier is {round(logistic_f1*100,2)}%')


# In[296]:


# ROC curve and AUC 
probs = model2.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
log_auc = roc_auc_score(y_test, probs)

# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot curve
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.title(f"AUC = {round(log_auc,3)}")
plt.show()


# In[284]:


import pickle

# save the model to disk
filename = 'DecisionTreeClassifier.sav'
pickle.dump(model1, open(filename, 'wb'))
 


# In[ ]:





# # RandomForestClassifier

# In[ ]:





# In[290]:


model3 = RandomForestClassifier(max_depth=6)
model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)
print(accuracy_score(y_test, y_pred3))
accuracy[str(model3)] = accuracy_score(y_test, y_pred3)*100


# In[291]:


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred3)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# In[292]:


print(classification_report(y_test,y_pred3))


# In[293]:


from sklearn.metrics import f1_score
from sklearn.metrics import recall_score,precision_score,classification_report,roc_auc_score,roc_curve

logistic_f1 = f1_score(y_test, y_pred3)
print(f'The f1 score for RandomForestClassifier is {round(logistic_f1*100,2)}%')


# In[294]:


# ROC curve and AUC 
probs = model3.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
log_auc = roc_auc_score(y_test, probs)

# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot curve
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.title(f"AUC = {round(log_auc,3)}")
plt.show()


# In[ ]:


import pickle

# save the model to disk
filename = 'RandomForestClassifier.sav'
pickle.dump(model1, open(filename, 'wb'))
 


# In[ ]:





# # GradientBoostingClassifier

# In[ ]:





# In[297]:


model4 = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1)
model4.fit(X_train, y_train)
y_pred4 = model4.predict(X_test)
print(accuracy_score(y_test, y_pred4))
accuracy[str(model4)] = accuracy_score(y_test, y_pred4)*100


# In[299]:


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred4)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# In[303]:


print(classification_report(y_test,y_pred4))


# In[304]:


from sklearn.metrics import f1_score
from sklearn.metrics import recall_score,precision_score,classification_report,roc_auc_score,roc_curve

logistic_f1 = f1_score(y_test, y_pred3)
print(f'The f1 score forGradientBoostingClassifier is {round(logistic_f1*100,2)}%')


# In[305]:


# ROC curve and AUC 
probs = model4.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
log_auc = roc_auc_score(y_test, probs)

# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot curve
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.title(f"AUC = {round(log_auc,3)}")
plt.show()


# In[306]:


import pickle

# save the model to disk
filename = 'GradientBoostingClassifier.sav'
pickle.dump(model1, open(filename, 'wb'))
 


# In[ ]:





# # Compare All Models with Accuracy Graph

# In[ ]:





# In[307]:


accuracy


# In[308]:


algos = list(accuracy.keys())
accu_val = list(accuracy.values())

plt.bar(algos, accu_val, width=0.4)
plt.title('Accuracy Differences')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.show()


# In[309]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import xgboost


# In[310]:


param_combinations = {
    'learning_rate': np.arange(0.05, 0.4, 0.05),
    'max_depth': np.arange(3, 10),
    'min_child_weight': np.arange(1, 7, 2),
    'gamma': np.arange(0.0, 0.5, 0.1),
}


# In[311]:


param_combinations = {
    'learning_rate': np.arange(0.05, 0.4, 0.05),
    'max_depth': np.arange(3, 10),
    'min_child_weight': np.arange(1, 7, 2),
    'gamma': np.arange(0.0, 0.5, 0.1),
}

XGB = xgboost.XGBClassifier()
perfect_params = RandomizedSearchCV(XGB, param_distributions=param_combinations, n_iter=6, n_jobs=-1, scoring='roc_auc')

perfect_params.fit(X, y)
perfect_params.best_params_


# In[312]:


model5 = xgboost.XGBClassifier(min_child_weight=3, max_depth=8, learning_rate=0.05, gamma=0.0)
score = cross_val_score(model5, X, y, cv=10)


# In[313]:


print(score)
print('Mean: ', score.mean())


# In[ ]:





# In[ ]:




