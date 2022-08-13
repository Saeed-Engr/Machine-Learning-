#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[3]:


#Reading the data file
df = pd.read_csv('D:/Machine Learning projects/datasets/airline_passenger_satisfaction/train.csv')
df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().any()


# In[8]:


df.isnull().sum()


# Some observations about the data:
# -
# - The average delay in flights are 15 minutes, with a deviation of 38
# 
# - Median of the delays are 0, which means 50% of the flights from this data, were not delayed
# 
# - The count of Female passengers are more with 52727 !!
Now, as we have conducted a preliminary analysis of the data, lets segregate the features into categorical and numerical. Before that let us remove the 'Id' and 'Unnamed: 0' feature from the data
# In[9]:


df.drop(['Unnamed: 0','id'], axis=1, inplace=True)


# In[10]:


cat_features = [feature for feature in df.columns if df[feature].dtypes == 'O']
print('Number of categorical variables: ', len(cat_features))
print('*'*80)
print('Categorical variables column name:',cat_features)


# In[11]:


numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
print('Number of numerical variables: ', len(numerical_features))
print('*'*80)
print('Numerical Variables Column: ',numerical_features)


# In[ ]:





# VISUALIZATION AND INSIGHTS FROM DATA
# -
# - We will visualize any specific trends in the features, and that would help us in feature selection,
# and better understanding of the data. Later, we will derive some useful insights from our custom made function!

# Visualizing Categorical Features
# -

# In[12]:


for col in cat_features[:-1]:
    plt.figure(figsize=(6,3), dpi=100)
    sns.countplot(data=df,x=col,hue ='satisfaction',palette='gist_rainbow_r')
    plt.legend(loc=(1.05,0.5))


# - Now it's time for some insights with percent values, to back up our conclusions above

# In[13]:


def get_insights(data):
    for cols in cat_features[:-1]:
        cat_group = data.groupby(cols)['satisfaction'].describe()
        percent = 100 *(cat_group['freq']/cat_group['count'])
        print('*'*70)
        print("Insights from '{}' data".format(cols))
        print('*'*70)
        for i in range(0,(len(cat_group))):
            print('{} from {} are {}'.format(round(percent[i],2),percent.index[i], cat_group['top'][i]))
        print('*'*70)


# In[14]:


get_insights(df)


# From the visualizations and insights, some points are clear:
# -
# 
# - Gender doesn't play an important role in the satisfaction, as men and women seems to equally concerned about the same factors
# 
# - Number of loyal customers for this airline is high, however, the dissatisfaction level is high irrespective of the loyalty. Airline will have to work on maintaining the loyal customers
# 
# - Business Travellers seems to be more satisfied with the flight, than the personal travellers
# 
# - People in business class seems to be the most satisfied lot, and those in economy class are least satisfied

# In[15]:


#Creating a heatmap of the correlation values
sns.heatmap(df.corr())


# In[16]:


#Plotting the barplot of numerical features
for col in numerical_features:
    plt.figure(figsize=(6,3), dpi=100)
    sns.barplot(data=df,x='satisfaction',y=col,palette='gist_rainbow_r')


# In[ ]:





# - From the plots, it is clear that age and Gate location, does not play a huge role in flight satisfaction,
# - and also the gender does not tell us much as seen in the earlier plot.Hence we drop these values

# In[17]:


#Dropping age, gender and gate location
df.drop(['Age','Gender','Gate location'], axis=1, inplace=True)


# - Let us focus now on the most important factor, we discussed ealier, which is the flight delays. Let us do an analysis of the delays,
# and it's relation with the satisfaction

# In[18]:


df.groupby('satisfaction')['Arrival Delay in Minutes'].mean()


# In[19]:


plt.figure(figsize=(10,5), dpi=100)
sns.scatterplot(data=df,x='Arrival Delay in Minutes',y='Departure Delay in Minutes',hue='satisfaction',palette='gist_rainbow_r', alpha=0.8)


# The most important takeaway here is the longer the flight distance, most passengers are okay with a slight delay in departure, which is a strange finding from this plot! So departure delay is less of a factor for a long distance flight, comparitively, however, short distance travellers does not seem to be excited about the departure delays, which also makes sense
# 
# Generally, business class seems to have been satisfied more than the passengers from economy or economy plus, let's analyze that

# In[20]:


df.groupby('Class').mean()


# In[ ]:





# So, the people from business class have given higher ratings for all the services provided, compared to Eco and Eco plus. Hence 'class' of travel should be a big factor in satisfaction

# MISSING VALUES
# -
# We have 310 missing values in 'Arrival delay', which is not quiet big, in terms of the data point we have, however, we'd not drop these values. And also we'd not settle with a strategy to calculate the mean, and fill in the missing values. We'd try something different here
# 
# We are already aware that the 'Arrival time' and 'Departure Time' have a sort of linear relationship, so we'll substitute the same departure delay values to the arrival delay values, for the missing values in data points

# In[21]:


#Creating a copy of the dataset, before we delete the NA values and substitute
df_copy=df.copy()


# In[22]:


df.isna().sum()


# Now we have handled the missing data!
# -

# In[23]:


#Let's try plotting the new values
sns.scatterplot(data=df_na, x='Arrival Delay in Minutes', y='Departure Delay in Minutes')


# MAPPING THE CATEGORICAL VARIABLES
# -
# We will map the binary variables with 0 and 1, and we will use the get_dummies method for the remaining variables

# In[25]:


#Mapping the values
df['satisfaction'] = df['satisfaction'].map({'neutral or dissatisfied':0 , 'satisfied':1})
df['Customer Type'] = df['Customer Type'].map({'Loyal Customer':1, 'disloyal Customer':0})
df['Type of Travel'] = df['Type of Travel'].map({'Personal Travel':0, 'Business travel':1})


# In[26]:


#Transforming the dataframe with mapped values
df=pd.get_dummies(df)


# In[27]:


#Checking the data
df.head()


# In[ ]:





# BUILDING THE MODEL
# -
# Now comes the most exciting part of building a model wiwth our data, and let's do some predictions, and most importantly find out the features that stand out

# In[28]:


#Preparing X and Y
X = df.drop('satisfaction', axis=1)
y = df['satisfaction']


# In[37]:


#Importing our ML toolkit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, plot_confusion_matrix,classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


# In[38]:


#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[44]:


#Scaling the features with pipeline method, and standard scaler
pipeline = Pipeline([
            ('std_scaler',StandardScaler()),
                    ])
scaled_X_train = pipeline.fit_transform(X_train)
scaled_X_test = pipeline.transform(X_test)


# In[45]:


kfold = StratifiedKFold(n_splits=10)


# In[46]:


# Modeling step to test differents algorithms 
random_state = 2
classifiers = []
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(XGBClassifier(random_state=random_state))

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, scaled_X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["LogisticRegression","KNeighbors","SVC","DecisionTree","AdaBoost",
"RandomForest","GradientBoosting","XGBoost"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


# In[ ]:





# From the curve, XGBoost works the best, and let us see the accuracy score, and other metrics
# -

# In[47]:


#Fitting the model to calculate accuracy
model_xgb = XGBClassifier(random_state =random_state)
model_xgb.fit(scaled_X_train,y_train)


# In[48]:


#Predicting and calculating accuracy score
pred_xgb = model_xgb.predict(scaled_X_test)
accuracy_score(y_test,pred_xgb)


# We get an accuracy score of 96%
# -

# In[50]:


plot_confusion_matrix(model_xgb,scaled_X_test,y_test)


# In[52]:


print(classification_report(y_test,pred_xgb))


# In[ ]:





# Now, let us plot the feature importances, and let us visualize it
# -

# In[53]:


orig_feature_names = X_train.columns
feature_important = model_xgb.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score")#, ascending=False)
ax =data.plot(kind='barh', figsize = (20,10))
ax.set_yticklabels(orig_feature_names); ## plot top features
#ax.set_xlabel("F-Score")
ax.set(xlabel="F-Score", ylabel="y label")
ax.set_title('Feature Importance')


# In[ ]:





# Clearly the most important features are:
# -
# Class of travel
# 
# Arrival/Departure delays
# 
# Services provided in the flight
# 
# Hence, our original hypothesis turned out true, as these are the golden factors that affect the passenger satisfaction

# In[ ]:




