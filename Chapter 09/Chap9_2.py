#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  
ranstat = np.random.RandomState(1)
P = np.dot(ranstat.rand(2, 2), ranstat.randn(2, 600)).T
plt.scatter(P[:, 0], P[:, 1])
plt.axis('equal');


# In[ ]:





# In[ ]:




