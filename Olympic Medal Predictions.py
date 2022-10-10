#!/usr/bin/env python
# coding: utf-8

# Steps followed for Linear Regression Modeling 
# 1. Form a hypothesis: We can predict how many medals a country will win in the Olympics.
# 2. find the Data: Data from the summer olympics
# 3. Reshape the Data 
# 4. Clean the Data to handle missing values
# 5. Error Metric (mean absolute error) add up error values and divide by total number of predictions
# 6. Splitting the Data: Train on 1 part, predict on another part.
# 7. Train a Model using linear regression using 2 predictors.

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np


# In[2]:


teams = pd.read_csv('teams.csv')


# In[3]:


teams


# In[4]:


#drop a couple columns

teams = teams[["team", "country",'year', 'athletes', 'age', 'prev_medals', 'medals']]
teams


# In[5]:


# looking for correlations with medals (athletes and prev medals is very high)
teams.corr()['medals']


# In[6]:

#plotting data with a regression line
sns.lmplot(x="athletes", y='medals', data=teams, fit_reg=True, ci=None) #ci is confidence interval


# In[7]:

# no relationship between age and medals
sns.lmplot(x="age", y='medals', data=teams, fit_reg=True, ci=None)


# In[8]:

#How many countries fall within each bin
teams.plot.hist(y='medals')


# In[9]:

### 4. Clean the Data to handle missing values
# finding the rows with missing values
teams[teams.isnull().any(axis=1)]


# In[10]:

#drop rows with missing data

teams = teams.dropna()
teams


# In[11]:

### 6. Splitting the Data: Train on 1 part, predict on another part.
# Last 2 years in test set, previous year into train set

train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()


# In[12]:

train.shape


# In[13]:

test.shape


# In[14]:

### 7. Train a Model using linear regression using 2 predictors.

reg = LinearRegression()


# In[15]:

#columns we are going to use to predict
predictors = ['athletes', 'prev_medals']
target = 'medals' #to predict this column


# In[16]:

# data we will use, followed by the target
reg.fit(train[predictors], train['medals'])


# In[17]:

#using alg to make predictions
predictions = reg.predict(test[predictors])

# In[18]:

predictions


# In[19]:

# correcting the model to prevent negatives and rounding the numbers
# assigning the column to the test set
test['predictions'] = predictions
test

# In[20]:

# locate negative numbers and turn them into a 0
test.loc[test['predictions']<0, 'predictions']=0

# In[21]:

#rounding predictions to nearest whole number
test['predictions'] = test['predictions'].round()


# In[22]:

test
# In[23]:

# looking at mean absolute error

error = mean_absolute_error(test['medals'], test['predictions'])

error

# about 3.3 medals away from the actual count

# In[24]:

#comparing and making sure our error is below the standard deviation
teams.describe()['medals']

# In[25]:

#looking at a specific country 
test[test['team']=='USA']


# In[26]:

#looking at a specific country 
test[test['team']=='IND']


# In[27]:

# errors by country

errors = (test['medals'] - test['predictions']).abs()


# In[28]:

errors


# In[29]:

# seperate group for each team and then find the mean
error_by_team = errors.groupby(test['team']).mean()


# In[30]:

error_by_team

# In[31]:

#medals each country earned on average
medals_by_team = test['medals'].groupby(test['team']).mean()

# In[32]:

medals_by_team

# In[33]:

#ratio of error
error_ratio = error_by_team / medals_by_team

# In[34]:

error_ratio

# In[35]:

#countries that dont have missing values
error_ratio[~pd.isnull(error_ratio)]

# In[36]:

#clean up infinite values

error_ratio = error_ratio[np.isfinite(error_ratio)]


# In[37]:

error_ratio

# In[38]:

### USA within 12%
error_ratio.plot.hist()


# In[39]:

### making predictions for countries that earn alot of medals, this models works well
### countries that do not get a lot of medals the error ration tends to be very high

error_ratio.sort_values()


# to improve accuacy
## 1 add more predictions
## 2 try different models
## 3 go back to the original athlete data set
## 4 try reshaping the columns
## 5 measure the error across more columns
## 6 measure across different country parameters

