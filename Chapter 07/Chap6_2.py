#!/usr/bin/env python
# coding: utf-8

# In[14]:



import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set() 
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.60)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='autumn');
fit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='autumn')
plt.plot([0.6], [2.1], 'x', color='green', markeredgewidth=2, markersize=10)
for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(fit, m * fit + b, '-k')
plt.xlim(-1, 3.5);


# In[ ]:





# In[ ]:





# 

# In[ ]:




