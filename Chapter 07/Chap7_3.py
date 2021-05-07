#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
P = np.array([[-1, 2], [1, -1], [2, 1], [2, 2]])
Q = np.array([1, 2, 1, 1])
SGDClassif = linear_model.SGDClassifier(max_iter = 1000, tol=1e-3,penalty = "elasticnet")
SGDClassif.fit(P, Q)
SGDClassifier(alpha=0.0001, average=False, class_weight=None,
              early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
              l1_ratio=0.15, learning_rate='optimal', loss='hinge',
              max_iter=1000, n_iter_no_change=5, n_jobs=None,
              penalty='elasticnet', power_t=0.5, random_state=None,
              shuffle=True, tol=0.001, validation_fraction=0.1, verbose=0,
              warm_start=False)
SGDClassif.predict([[3.,3.]])
SGDClassif.coef_
SGDClassif.intercept_
SGDClassif.decision_function([[3., 3.]])

import numpy as np
from sklearn import linear_model
nsamples, nfeatures = 9, 4
rng = np.random.RandomState(0)
q = rng.randn(nsamples)
p = rng.randn(nsamples, nfeatures)
SGDReg =linear_model.SGDRegressor(
   max_iter = 1000,penalty = "elasticnet",loss = 'huber',tol = 1e-3, average = True
)
SGDReg.fit(p, q)
SGDRegressor(alpha=0.0001, average=True, early_stopping=False, epsilon=0.1,
             eta0=0.01, fit_intercept=True, l1_ratio=0.15,
             learning_rate='invscaling', loss='huber', max_iter=1000,
             n_iter_no_change=5, penalty='elasticnet', power_t=0.25,
             random_state=None, shuffle=True, tol=0.001,
             validation_fraction=0.1, verbose=0, warm_start=False)
SGDReg.coef_
SGDReg.intercept_
SGDReg.t_


# In[ ]:




