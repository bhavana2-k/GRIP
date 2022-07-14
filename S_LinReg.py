#!/usr/bin/env python
# coding: utf-8

# # Data Science And Business Analytics Intern 
# **The Spark Foundation** Task #1
# 
# **Prediction using Supervised ML**
# 
# **Name : Bhavana Khairnar**

# In this regression task, we will predict the percentage of marks that the student is expected to score based on the number of hours they studied. It is the simple linear regression task and contains only two variables.

# In[19]:


# Importing all libraries required 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[20]:


# Reading data from remote link

url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)


# In[22]:


# Plotting the data for visualization
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[23]:


# Preparing the data
X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values 


# In[24]:


# Splitting the data in training and test sets.
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# In[25]:


#Algorithm
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[26]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[27]:


# Predictions
print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[28]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[29]:


# Verification with original data
hours = 9.25
own_pred = regressor.predict([[hours]])
print(own_pred)


# **If student studies for 9.25 hours, then the estimated or expected percentage of the student is 93.69**
# 
# 

# In[30]:


#Evaluaiting the model in terms of accuracy
from sklearn import metrics
print("R_squared",metrics.r2_score(y_test,y_pred))
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# The model accuracy os 94.55% and the Mean Avbsolute Error is 4.184

# **Thank You**
# 
