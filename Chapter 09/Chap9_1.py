#!/usr/bin/env python
# coding: utf-8

# In[7]:


print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets

np.random.seed(5)

irisdata = datasets.load_iris()
P = irisdata.data
q = irisdata.target
estimatorskmeans = [('k_means_iris_8', KMeans(n_clusters=8)),
              ('k_means_iris_3', KMeans(n_clusters=3)),
              ('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1,
                                               init='random'))]
fignum=1
titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
for name, est in estimatorskmeans:
    fig = plt.figure(fignum,figsize=(4,3))
    px = Axes3D(fig, rect=[0, 0, .82,1], elev=52,azim=147)
    est.fit(P)
    labels = est.labels_

    px.scatter(P[:, 3], P[:, 0], P[:, 2],
               c=labels.astype(np.float), edgecolor='k')

    px.w_xaxis.set_ticklabels([])
    px.w_yaxis.set_ticklabels([])
    px.w_zaxis.set_ticklabels([])
    px.set_xlabel('Petal width')
    px.set_ylabel('Sepal length')
    px.set_zlabel('Petal length')
    
    px.dist = 12
    fignum=fignum+1

fig = plt.figure(fignum, figsize=(4, 3))
px = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

for name, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
    px.text3D(P[q == label, 3].mean(),
              P[q == label, 0].mean(),
              P[q == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
q= np.choose(q,[1, 3, 0]).astype(np.float)
px.scatter(P[:, 3], P[:, 0], P[:, 2], c=q, edgecolor='k')

px.w_xaxis.set_ticklabels([])
px.w_yaxis.set_ticklabels([])
px.w_zaxis.set_ticklabels([])
px.set_xlabel('Petal width')
px.set_ylabel('Sepal length')
px.set_zlabel('Petal length')
px.set_title('Ground Truth')
px.dist = 12

fig.show()


# In[ ]:





# In[ ]:




