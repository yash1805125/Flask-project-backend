#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import plotly.offline as py
import seaborn as sns
import pickle
py.init_notebook_mode(connected=True)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression


# In[6]:


# Data import
attrition = pd.read_csv('emp-data.csv')
attrition.head()
# attritionnum = pd.read_csv('C:/Users/KIIT/Desktop/emp_att_f.csv')
# attritionnum.head()


# In[2]:


df = pd.DataFrame(data=attrition, columns = ['EmployeeNumber', 'Age', 
        'BusinessTravel', 'MonthlyIncome', 'MonthlyRate', 'Department', 'DistanceFromHome',
        'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'DailyRate', 'JobInvolvement',
       'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'NumCompaniesWorked',
       'OverTime', 'StandardHours', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 
       'StockOptionLevel','TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
       'Attrition'])
df.head()



# In[4]:


df["Attrition"].value_counts()


# In[5]:


df["Attrition"].value_counts().plot(kind="bar", color=["Lime", "Red"]);


# In[6]:


df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
df.head()


# In[7]:


del df['EmployeeNumber']
df.head()


# In[8]:


# Create two new dataframes, one with the training rows, one with the test rows
train, test = df[df['is_train']==True], df[df['is_train']==False]


# In[9]:


# Create a list of the feature column's names - without 33rd column attrition
features = df.columns[:31]


# In[10]:


# Defining the variable to be predicted - the target
x_train = train
y_train = train['Attrition']
train.head()
len(train)


# In[11]:


x_test = test
y_test = test['Attrition']
test.head()


# In[16]:


# Create a random forest Classifier
clf = RandomForestClassifier(n_estimators=25, n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate to the training y (attrition_num)
clf.fit(train[features], y_train)

# Apply the Classifier we trained to the test data
clf.predict(test[features])

# View the predicted probabilities of the first 10 observations
clf.predict_proba(test[features])[0:10]


# In[17]:


clf.score(test[features],y_test)


# In[19]:


preds = clf.predict(test[features])


# In[31]:


# Create confusion matrix, WHICH REALLY IS CONFUSING AT FIRST
pd.crosstab(test['Attrition'], preds, rownames=['Actual Attrition'], colnames=['Predicted Attrition'])


# In[20]:


log_reg = LogisticRegression(n_jobs=2, random_state=0)


# In[33]:


log_reg.fit(train[features], y_train)


# In[34]:


log_reg.fit(train[features], y_train)


# In[21]:


# Apply the Classifier we trained to the test data
log_reg.predict(test[features])

# View the predicted probabilities of the first 10 observations
log_reg.predict_proba(test[features])[0:10]


# In[35]:


log_reg.score(test[features],y_test)


# In[22]:


#Create a Hyperparameter grid for Logistic Regression
log_reg_grid = {"C": np.logspace(-4,4,20),
               "solver": ["liblinear"]}

#Create Hyperparameter grid for Random Forest\
rf_grid = {"n_estimators": np.arange(100,1000,50),
          "max_depth": [None,3,5,10],
          "min_samples_split": np.arange(2,20,2),
          "min_samples_leaf": np.arange(1,20,2)}



# In[23]:


#Tune Logistic Regression

np.random.seed(42)

#Setup random Hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                               param_distributions=log_reg_grid,
                               cv=5,
                               n_iter=20,
                               verbose=True)
#Fit random hyperparameter search model for LogisticReession
rs_log_reg.fit(train[features], y_train)


# In[25]:


rs_log_reg.best_params_


# In[39]:


print(rs_log_reg.score(test[features],y_test))


# In[40]:




# In[26]:


# Apply the Classifier we trained to the test data
# rs_log_reg.predict(test[features])

# # View the predicted probabilities of the first 10 observations
# rs_log_reg.predict_proba(test[features])[0:10]


# # In[28]:


# p = rs_log_reg.predict(test[features])


# # In[42]:


# # Create confusion matrix, WHICH REALLY IS CONFUSING AT FIRST
# pd.crosstab(test['Attrition'], p, rownames=['Actual Attrition'], colnames=['Predicted Attrition'])



# # In[ ]:

pickle.dump(rs_log_reg, open('model_1.pkl','wb'))




