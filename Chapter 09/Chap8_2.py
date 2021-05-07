#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
iris = load_iris()
decisiontree = DecisionTreeClassifier(random_state=0, max_depth=2)
decisiontree = decisiontree.fit(iris.data, iris.target)
er = export_text(decisiontree, feature_names=iris['feature_names'])
print(er)


# In[ ]:





# In[ ]:




