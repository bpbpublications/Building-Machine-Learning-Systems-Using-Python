#!/usr/bin/env python
# coding: utf-8

# In[17]:



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import __future__
df=pd.read_csv("D:\\Drive C\\Desktop\\bookbpb\\umbrella.csv") # Data is uploaded and a DataFrame is
#created using Pandas pd object


# In[18]:


df


# In[19]:


X = np.asarray(df.Rainfall.values)
y = np.asarray(df.UmbrellasSold.values)
#Scaling and Normalization of features is performed
def FeatureScalingNormalization(X):
# Xnorm is a copy of X vector
    Xnorm = X
 # avgx will contain average value of x in training set
    avgx = np.zeros(X.shape[0])
 #rangex will contain standard deviation values of x
    rangex = np.zeros(X.shape[0])
    avgx = X.mean()
    rangex = X.std(ddof=1) # Calculated with NumPy. It requires degreeoffreedom=1
 # No. of training examples is stored in p
    p = X.shape[0]
 # a vector of size p with the average values of x
    avgx_matrix = np.multiply(np.ones(p), avgx).T
 # a vector of size p with the standard deviation values
    rangex_matrix = np.multiply(np.ones(p), rangex).T
 # Normalization is applied on x values
    Xnorm = np.subtract(X, avgx).T
    Xnorm = Xnorm /rangex.T
    return [Xnorm, avgx, rangex]
featuresNormalizeresults = FeatureScalingNormalization(X)
# normalized X matrix is obtained
X = np.asarray(featuresNormalizeresults[0]).T
# mean values are obtained
avgx = featuresNormalizeresults[1]
# standard deviation values is obtained
rangex = featuresNormalizeresults[2]
X
p = len(y) # number of training examples
X = np.vstack((np.ones(p), X.T)).T # Training Examples, column of 1’s is added to X
X
plt.scatter(X[:,[1]], y, color='blue') # Data is plotted, Scatter plot is obtained
plt.xlabel("Rainfall")
plt.ylabel("Umbrellas Sold")
# We calculate the plot when two parameters, Theta is randomly chosen as [140.0,5.0]
theta_0 = 140.0
theta_1 = 5.0
theta = np.asarray([theta_0,theta_1]).astype(float)
# Plot the data
plt.scatter(X[:,[1]], y, color='black')
# corresponding to Hypothesis model, red line is plotted.
plt.plot(X[:,[1]], np.sum(np.multiply(X,theta), axis=1), color='red', linewidth=1)
plt.xlabel("Rainfall")
plt.ylabel("Umbrellas Sold")
# Calculate Cost Function using 3 values in a Dataset. We have taken random values of theta as [120.0, 10.0].
X1=X[0:3]
y1=y[0:3]
m1=3
theta_0 = 120.0
theta_1 = 10.0
theta = np.asarray([theta_0,theta_1]).astype(float)
plt.scatter(X1[:,[1]], y1, color='blue')
plt.plot(X1[:,[1]], np.sum(np.multiply(X1,theta), axis=1), color='red', linewidth=1)
# Plot red points corresponding to the predicted values.
plt.scatter(X1[:,[1]], np.sum(np.multiply(X1,theta), axis=1), color='red')
plt.xlabel("Rainfall")
plt.ylabel("Umbrella Sold")
# Cost Function is Calculated
def calcCostFunction(X, y, theta):
 # number of training examples
    p = len(y)
 # Cost J is initialized
    J = 0
 # Calculate h = X * theta
    h = np.sum(np.multiply(X, theta), axis=1)
 # Squared Error = (h - y)^2 (vectorized)
    SquaredError = np.power(np.subtract(h,y), 2)
 # Calculate the Cost J
    J = 1/(2*p) * np.sum(SquaredError)
    return J
calcCostFunction(X,y,theta)
import random # import the random library 
print("[Th0 Th1]", "\tJ")
for x in range(10):
    theta_0 = random.randint(1,101)
    theta_1 = random.randint(1,101)
    theta = np.asarray([theta_0, theta_1]).astype(float)
 # Calculate J and print the table
    print(theta, calcCostFunction(X, y, theta))


# In[ ]:





# In[ ]:





# In[ ]:




