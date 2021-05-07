#!/usr/bin/env python
# coding: utf-8

# In[9]:


from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.preprocessing import StandardScaler
ssc = StandardScaler()
ssc.fit(X_train)
X_train_std = ssc.transform(X_train)
X_test_std = ssc.transform(X_test)
from sklearn.linear_model import LogisticRegression
weights, params = [], []
for c in np.arange(1, 5):
    lrr = LogisticRegression(C=10**c, random_state=0)
    lrr.fit(X_train_std, y_train)
    weights.append(lrr.coef_[1])
    params.append(10**c)
weights = np.array(weights)
# Decision region drawing
import matplotlib.pyplot as plt
plt.plot(params, weights[:, 0], color='blue', marker='x', label='petal length')
plt.plot(params, weights[:, 1], color='green', marker='o', label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='right')
plt.xscale('log')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




