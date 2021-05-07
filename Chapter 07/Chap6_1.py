#!/usr/bin/env python
# coding: utf-8

# In[10]:



import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set() 
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.60)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='autumn');


# In[ ]:





# In[ ]:





# 
