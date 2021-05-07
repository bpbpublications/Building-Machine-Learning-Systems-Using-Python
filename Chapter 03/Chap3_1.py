#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# Loading the predefined iris dataset from Sklearn
data = load_breast_cancer()
# Organizing the data
labelnames = data['target_names']
labels = data['target']
featurenames = data['feature_names']
features = data['data']
# data
print(labelnames)
print('Class label = ', labels[0])
print(featurenames)
print(features[0])
# Splitting the data
train, test, train_labels, test_labels = train_test_split(features,labels,test_size=0.33,random_state=42)
# Initializing the classifier
gnb = GaussianNB()
# Training classifier
model = gnb.fit(train, train_labels)
# Making predictions
prediction = gnb.predict(test)
print(prediction)
# Evaluating the accuracy
print(accuracy_score(test_labels, prediction))


# In[ ]:




