#!/usr/bin/env python
# coding: utf-8

# In[15]:



import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set() 
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.60)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='autumn');
fit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='autumn')
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    fit2 = m * fit + b
    plt.plot(fit, fit2, '-k')
    plt.fill_between(fit, fit2 - d, fit2 + d, edgecolor='none',
                     color='#AAAAAA', alpha=0.4)
plt.xlim(-1, 3.5);


# In[ ]:





# In[ ]:





# 

# In[ ]:




