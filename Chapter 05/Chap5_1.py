#!/usr/bin/env python
# coding: utf-8

# In[20]:


print(__doc__)
import numpy as np
from sklearn import linear_model
train_samples, test_samples, n_features = 80, 200, 700
np.random.seed(0)
coefficient = np.random.randn(n_features)
coefficient[50:] = 0.0
X = np.random.randn(train_samples + test_samples, n_features)
y = np.dot(X, coefficient)
# Splitting train and test data
train_X, test_X = X[:train_samples], X[train_samples:]
train_y, test_y = y[:train_samples], y[train_samples:]
# Finding train and test errors
alphas = np.logspace(-5, 1, 60)
enet = linear_model.ElasticNet(l1_ratio=0.7, max_iter=10000)
train_errors = list()
test_errors = list()
for alpha in alphas:
 enet.set_params(alpha=alpha)
 enet.fit(train_X, train_y)
 train_errors.append(enet.score(train_X, train_y))
 test_errors.append(enet.score(test_X, test_y))
i_alpha_optimum = np.argmax(test_errors)
alphaoptimum = alphas[i_alpha_optimum]
print("Optimal regularization parameter is: %s" % alphaoptimum)
# Finding the coefficient on full data with optimal regularization parameter
enet.set_params(alpha=alphaoptimum)
coefficient1 = enet.fit(X, y).coef_
# Plotting results functions
import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)
plt.semilogx(alphas, train_errors, label='Train')
plt.semilogx(alphas, test_errors, label='Test')
plt.vlines(alphaoptimum, plt.ylim()[0], np.max(test_errors), color='k',
linewidth=3, label='Optimum on test')
plt.legend(loc='lower left')
plt.ylim([0, 1.2])
plt.xlabel('Regularization parameter')
plt.ylabel('Performance')
# Show estimated coef_ versus true coef
plt.subplot(2, 1, 2)
plt.plot(coefficient, label='True coef')
plt.plot(coefficient1, label='Estimated coef')
plt.legend()
plt.subplots_adjust(0.07, 0.05, 0.87, 0.87, 0.22, 0.29)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




