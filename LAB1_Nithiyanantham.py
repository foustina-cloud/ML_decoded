#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the necessary Python libraries.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Reading the data from the .csv file using a Pandas framework.
data = pd.read_csv('student_marks.csv')

#Creating arrays for the feature and target variables.
X = np.array(data['Midterm mark'].values)
Y = np.array(data['Final mark'].values)

#Plotting the data as a scatter plot to visualize the data.
plt.scatter(X, Y)
plt.ylabel("Final Marks")
plt.xlabel("Midterm Marks")
plt.title("Students' Midterm and Final marks")
plt.show()


# In[2]:


#The feature variable 'X' is standardized and the new variable 'X_scaled' is generated in an array.
X_scaled = preprocessing.scale(X)

#Read the X_scaled array.
X_scaled


# In[3]:


#The feature variable 'Y' is standardized and the new variable 'Y_scaled' is generated in an array.
Y_scaled = preprocessing.scale(Y)

#Read the Y_scaled array
Y_scaled


# In[4]:


#The standardized variables are graphed into a scatter plot to visualize and ensure standardization occurred 
#with a mean of 0 and unit variance.
plt.scatter(X_scaled, Y_scaled)
plt.ylabel("Final Marks (standardized)")
plt.xlabel("Midterm Marks (standardized)")
plt.title("Students' Midterm and Final marks (standardized)")
plt.show()


# In[10]:


# Building a Gradient Descent model to find the predicted linear regression.
m = -0.5 #The variable 'm' denotes the slope of the linear function (Theta1, as we learned it in class).
b = 0 #The variable 'b' denotes the y-intercept of the linear function (Theta0, as we learned it in class).

L = 0.0001 # The learning Rate
epochs = 100  # The number of iterations to perform gradient descent

n = float(len(X_scaled)) # Number of elements in X

# Performing Gradient Descent 
for i in range(epochs): 

    Y_pred = m * X_scaled + b #The predicted value of Y currently.
    E = (1/n) * sum(Y_scaled - Y_pred)**2 #The cost function modeled in this equation.
    D_m = (2/n) * sum(X_scaled * (Y_scaled - Y_pred))  # The partial derivate with respect to variable m.
    D_b = (2/n) * sum(Y_scaled - Y_pred)  # The partial derivative with respect to variable b.
    m = m - L * D_m  # Update m
    b = b - L * D_b  # Update b

    #Graphing the cost function with respect to variable 'm'
    plt.scatter(m, E)
    plt.ylabel("Cost")
    plt.xlabel("Feature variable 'm'")
    plt.title("The Error Function with respect with variable 'm'")
    plt.plot(m, E)
    
    #Graphing the cost function with respect to variable 'b'. This graph can be created by removing the comment hashtags from the next lines and adding comment hashtags to the plotting of the cost function in relation to the 'm' variable.
    #plt.scatter(b, E)
    #plt.ylabel("Cost")
    #plt.xlabel("Feature variable 'b'")
    #plt.title("The Error Function with respect to variable 'b'")
    #plt.plot(b, E)
    


# In[12]:


# Modeling a linear regression using the predictions.
Y_pred = m*X_scaled + b

plt.scatter(X_scaled, Y_scaled)
plt.ylabel("Final Marks")
plt.xlabel("Midterm Marks")
plt.title("Students' Midterm and Final marks (standardized)")
plt.plot([min(X_scaled), max(X_scaled)], [min(Y_pred), max(Y_pred)], color='green') # predicted
plt.show()


# In[13]:


# Changing the learning rate to 0.1
m = -0.5
b = 0

L = 0.1 # The learning Rate is now much larger.
epochs = 100  # The number of iterations to perform gradient descent.

n = float(len(X_scaled)) # The number of elements in X.

# Performing Gradient Descent 
for i in range(epochs): 

    Y_pred = m * X_scaled + b  # The predicted value of Y currently.
    E = (1/n) * sum(Y_scaled - Y_pred)**2  #The cost function modeled in this equation.
    D_m = (2/n) * sum(X_scaled * (Y_scaled - Y_pred)) # The partial derivate with respect to variable m.
    D_b = (2/n) * sum(Y_scaled - Y_pred)  #The partial derivative with respect to variable b.
    m = m - L * D_m  # Update m
    b = b - L * D_b  # Update b

#Graph the Cost Function
    plt.scatter(m, E)
    plt.plot(m, E)
    


# In[14]:


#Making new predictions with the learning rate of 0.1

Y_pred = m*X_scaled + b

plt.scatter(X_scaled, Y_scaled)
plt.ylabel("Final Marks")
plt.xlabel("Midterm Marks")
plt.title("Students' Midterm and Final marks (standardized)")
plt.plot([min(X_scaled), max(X_scaled)], [min(Y_pred), max(Y_pred)], color='green') # predicted
plt.show()


# In[ ]:




