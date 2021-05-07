#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# Predefined Sklearn Wine dataset is loaded
data = load_wine()
# data is organized
labelnames = data['target_names']
labels = data['target']
featurenames = data['feature_names']
features = data['data']
# Look at our data
print(labelnames)
print('Class label = ', labels[0])
print(featurenames)
print(features[0])
# Splitting our data
train, test, train_labels, test_labels = train_test_split(features,labels,test_size=0.33,random_state=42)
# Initializing our classifier
gnb = GaussianNB()
# Training our classifier
model = gnb.fit(train, train_labels)
# Making predictions
prediction = gnb.predict(test)
print(prediction)
# Evaluate accuracy
print(accuracy_score(test_labels, prediction))


# In[ ]:




