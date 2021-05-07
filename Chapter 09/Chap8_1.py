#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn import tree
P = [[0, 0], [1, 1]]
Q = [0, 1]
DTclf = tree.DecisionTreeClassifier()
DTclf = DTclf.fit(P, Q)
DTclf.predict([[2., 2.]])

DTclf.predict_proba([[2., 2.]])

from sklearn.datasets import load_iris
from sklearn import tree
P, q = load_iris(return_X_y=True)
DTclf = tree.DecisionTreeClassifier()
DTclf = DTclf.fit(P, q)
tree.plot_tree(DTclf)


# In[ ]:





# In[ ]:




