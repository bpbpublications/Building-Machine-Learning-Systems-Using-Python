#!/usr/bin/env python
# coding: utf-8

# In[19]:


from sklearn.linear_model import Ridge
ridgereg = Ridge(alpha=0, normalize=True)
ridgereg.fit(X_train, y_train)
y_pred = ridgereg.predict(X_test)
# calculate R square value, MAE, MSE, RMSE
from sklearn import metrics
#print("R-Square Value",r2_score(y_test,y_pred))
print ("\nmean_absolute_error :",metrics.mean_absolute_error(y_test, y_pred))
print ("\nmean_squared_error : ",metrics.mean_squared_error(y_test, y_pred))
print ("\nroot_mean_squared_error : ",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:





# In[ ]:





# In[ ]:




