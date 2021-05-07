#!/usr/bin/env python
# coding: utf-8

# In[1]:


import operator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
np.random.seed(0)
x = 1 - 3 * np.random.normal(0, 1, 50)
y = x - 3 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 50)
# Data is transformed to include another axis
x = x[:, np.newaxis]
y = y[:, np.newaxis]
polyfeatures= PolynomialFeatures(degree=3)
xpoly = polyfeatures.fit_transform(x)
model = LinearRegression()
model.fit(xpoly, y)
ypoly_pred = model.predict(xpoly)
rmse = np.sqrt(mean_squared_error(y,ypoly_pred))
r2 = r2_score(y,ypoly_pred)
print(rmse)
print(r2)
plt.scatter(x, y, s=20)
# Values of x are sorted according to degree before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,ypoly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, ypoly_pred, color='m')
plt.show()


# In[ ]:




