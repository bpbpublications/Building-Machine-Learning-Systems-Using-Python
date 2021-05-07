#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn import tree
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
DTclf = tree.DecisionTreeRegressor()
DTclf = DTclf.fit(X, y)
DTclf.predict([[1, 1]])


# In[ ]:





# In[ ]:




