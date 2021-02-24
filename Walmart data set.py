#!/usr/bin/env python
# coding: utf-8

# In[48]:


## Import all the necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import datetime as dt
import seaborn as sns


# In[5]:


import os
print(os.listdir("/home/sunil/Documents/Projects/Walmart dataset"))


# In[6]:


## Load all the data sets

df_features = pd.read_csv('/home/sunil/Documents/Projects/Walmart dataset/features.csv')
df_train = pd.read_csv('/home/sunil/Documents/Projects/Walmart dataset/train.csv')
df_test = pd.read_csv('/home/sunil/Documents/Projects/Walmart dataset/test.csv')
df_store = pd.read_csv('/home/sunil/Documents/Projects/Walmart dataset/stores.csv')


# In[14]:


## Get the info on the data set

df_features.info()


# In[15]:


## Get the info on the data set

df_store.info()


# In[23]:


## Get the info on the data set

df_test.head()


# In[22]:


## Get the info on the data set

df_train.head()


# In[20]:


## Look for statistical discription

df_features.describe()


# In[21]:


## Look for statistical discription

df_store.describe()


# In[24]:


## Now you have a brief idea about how the data is distributed in the data set
## We know that we have tables with null values, i.e one with holidays. Therefore, 

df_features['IsHoliday'].value_counts(dropna=True)


# In[25]:


## Now convert the data column to standard datetime datatype so that we can perform time series operations
## To better understand Walmart - Store Sales Forecasting data, I will analyze its distribution and behavior. 
## Since this is a product design problem, I will conduct all my analysis considering the 
## product and the final result.

## Here, the main goal is to correctly predict Weekly_Sales values. 
## To do so, we need to consider the evaluation metric for this problem.

df_features['Date'] = pd.to_datetime(df_features['Date'])
df_train['Date'] = pd.to_datetime(df_train['Date'])
df_test['Date'] = pd.to_datetime(df_test['Date'])


# In[27]:


## Transform the IsHoliday column to binary format (Numerical format for easy formating)

df_features['IsHoliday'] = LabelEncoder().fit_transform(df_features['IsHoliday'])
df_train['IsHoliday'] = LabelEncoder().fit_transform(df_train['IsHoliday'])
df_test['IsHoliday'] = LabelEncoder().fit_transform(df_test['IsHoliday'])
df_store['Size']  = LabelEncoder().fit_transform(df_store['Size'] )
df_store['Type'] = LabelEncoder().fit_transform(df_store['Type'])

# The transformed column will look like this
df_test.head()


# In[28]:


## Combine the features and the train data set

df_store_feture = pd.merge(df_train,df_features,how='inner',on=['Store','Date','IsHoliday'])

df_store_feture_test = pd.merge(df_test,df_features,how='inner',on=['Store','Date','IsHoliday'])

print("Merged Train & Feature df : ",df_store_feture.shape[0])


# In[29]:


## Combine the store, features and the train data set

df_final = pd.merge(df_store_feture,df_train,how='inner')

df_final_test = pd.merge(df_store_feture_test,df_test,how='inner')

print("Merged Store,Train & Feature df : ",df_final.shape[0])


# In[34]:


df_final_test.isnull().sum()


# In[31]:


df_final.isnull().sum()


# In[38]:


## Replace all NA values with O

markdown = pd.DataFrame(SimpleImputer().fit_transform(df_final[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']]),columns=['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'])
markdown_test = pd.DataFrame(SimpleImputer().fit_transform(df_final_test[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']]),columns=['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'])

df = df_final.drop(['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'],axis=1)
df_test_1 = df_final_test.drop(['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'],axis=1)

df_final = pd.concat([df,markdown],axis=1)
df_final_test = pd.concat([df_test_1,markdown_test],axis=1)

df_final_test.isnull().sum()


# In[41]:


df_final_test.CPI.fillna(df_final_test.CPI.mean(),inplace=True)
df_final_test.Unemployment.fillna(df_final_test.Unemployment.mean(),inplace=True)


# In[43]:



df_final_test.head()


# In[44]:


df_grp = df_final[['Year','Dept','Weekly_Sales']].groupby(['Year','Dept']).mean().reset_index()
df_grp.head()


# In[54]:


plt.figure(figsize=(10,7))
sns.boxplot(data = df_grp,x= df_grp['Year'],y=df_grp['Weekly_Sales'])
plt.show()


# In[62]:


def scatter(dataset, column):
    plt.figure()
    plt.scatter(dataset[column] , dataset['Weekly_Sales'])
    plt.ylabel('Weekly_Sales')
    plt.xlabel(column)
    
scatter(df_final, 'Fuel_Price')
scatter(df_final, 'CPI')
scatter(df_final, 'IsHoliday')
scatter(df_final, 'Unemployment')
scatter(df_final, 'Temperature')
scatter(df_final, 'Store')
scatter(df_final, 'Dept')


# In[66]:


weekly_sales = df_final['Weekly_Sales'].groupby(df_final['Dept']).mean()
plt.figure(figsize=(25,8))
sns.barplot(weekly_sales.index, weekly_sales.values, palette='dark')
plt.grid()
plt.title('Average Sales - per Dept', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Dept', fontsize=16)
plt.show()


# In[67]:


weekly_sales = df_final['Weekly_Sales'].groupby(df_final['Year']).mean()
plt.figure(figsize=(25,8))
sns.barplot(weekly_sales.index, weekly_sales.values, palette='dark')
plt.grid()
plt.title('Average Sales - per Dept', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.show()


# In[68]:


weekly_sales = df_final['Weekly_Sales'].groupby(df_final['Month']).mean()
plt.figure(figsize=(25,8))
sns.barplot(weekly_sales.index, weekly_sales.values, palette='dark')
plt.grid()
plt.title('Average Sales - per Dept', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Month', fontsize=16)
plt.show()


# In[69]:


## Plotting the correlation matrix. Here I will check for relations with the Weekly Sales feature

plt.figure(figsize=(18,7))
sns.heatmap(df_final.corr(),annot=True)
plt.show()


# In[70]:


## Dropping features with lower correlation

df_final.drop(columns=['Month','Date'],inplace = True)
column_date = df_final_test['Date']
df_final_test.drop(columns=['Month','Date'],inplace = True)


# In[71]:


## Splitting the data set into two

X = df_final.drop('Weekly_Sales',axis=1)
y = df_final['Weekly_Sales']


# In[73]:


## Split data set into train and test

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.3,random_state=34)

print("X Train Shape :",X_train.shape)
print("X Val Shape   :",X_val.shape)
print("Y Train Shape :",y_train.shape)
print("Y Val Shape   :",y_val.shape)


# In[79]:


## The function plots the graph relation between a categorized feature and the Weekly_Sales

def graph_relation_to_weekly_sale(col_relation, df, x='Week', palette=None):
    df.Date = pd.to_datetime(df.Date)
    df['Week'] = df.Date.dt.week
    df['Month'] = df.Date.dt.month
    df['Year'] = df.Date.dt.year
    
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.relplot(
        x=x,
        y='Weekly_Sales',
        hue=col_relation,
        data=df,
        kind='line',
        height=5,
        aspect=2,
        palette=palette
    )
    plt.show()


# In[80]:


## Here we can see a seasonal behavior on the sales

graph_relation_to_weekly_sale('Year', df_train, x='Date', palette='Set2')


# In[81]:


# Week of the year can explain some Weekly_Sales variation
graph_relation_to_weekly_sale(None, df_train, x='Week')


# In[82]:


## The current year can also explain some Weekly_Sales variation since the weekly sales seem to be decreasing 
## with time. However, we need to be careful before jumping to conclusions. The last year (2012) does not have 
## Thanksgiving and Christmas days on the training set. The average sales can also be low because of that.

graph_relation_to_weekly_sale(None, df_train, x='Year')


# In[85]:


# Linear Regression model

lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_val)
lr_rmse_score = np.sqrt(mean_squared_error(y_pred,y_val))
lr_r2_score = r2_score(y_pred,y_val)
print("Root Mean Squared Error :",lr_rmse_score)
print("R2Score                 :",lr_r2_score)


# In[86]:


# Decission tree

dt = DecisionTreeRegressor()
dt_model=dt.fit(X_train,y_train)         
y_pred_dtone=dt_model.predict(X_val) 

## calculate RMSE

rms_dt = np.sqrt(mean_squared_error(y_pred_dtone,y_val))
r2_dt = r2_score(y_val, y_pred_dtone)
print('RMSE of Decision Tree Regression:',rms_dt)
print('R-Squared value:',r2_dt)
R2 = r2_score(y_val, y_pred)
n = X_train.shape[0]
p = len(X_train.columns)
Adj_r2 = 1-(1-R2)*(n-1)/(n-p-1)
print('Adjusted R-Square is : ',Adj_r2)


# In[ ]:


# Random Forest

rf_reg = RandomForestRegressor()

rf_model = rf_reg.fit(X_train,y_train)          
y_pred_rf = rf_model.predict(X_val)

rmse_rf = np.sqrt(mean_squared_error(y_pred_rf,y_val))
r2_rf = r2_score(y_pred_rf,y_val)

print('RMSE of predicted in RF model:',rmse_rf)
print('R Sqaured in RF model        :',r2_rf)


# In[ ]:



## Work with time series decomposition methods, to get a better view of seasonal components and get other 
## important data markdowns besides the four holidays that Walmart provided;

## The created custom features, besides time-related ones, don't seem to contribute to this model. 
## When we look at the Feature Importance, HolidayType didn't have a great contribution. 
## Maybe this is happening because this feature and the feature Week kind of explain the same thing. 
## A further investigation in this matter would be a nice next step;

## Try new models and hyperparameters combinations.

