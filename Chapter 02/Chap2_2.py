#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import sys
get_ipython().system('{sys.executable} -m pip install sklearn')
get_ipython().system('{sys.executable} -m pip install statsmodels')
from sklearn import linear_model

import statsmodels.api as sm
Stock_Market = {'Year':[2018,2019,2018,2017,2017,2016,2017,2019,2018,2018,2019,2019,2016,2017,2017,2018,2018,2018,2018,2018,2016,2016,2016,2016], 'Month': [10, 12,10,9,8,4,6,5,7,5,1,2,12,10,11,12,8,9,6,4,5,1,3,2], 'Rateofinterest':
[3.25,4.5,3.5,5.5,4.5,4.5,3.5,5.25,6.25,4.25,5,5,6,5.75,4.75,5.75,5.75,4.75,5.75,4.75,3.75,4.75,5.75,5.75],
'RateofUnemployment':[7.3,4.3,3.3,4.3,6.4,5.6,4.5,5.5,6.5,4.6,6.7,5.9,6,4.9,6.8,5.1,7.2,5.1,6.1,7.1,6.9,5.2,7.2,5.1],'Stock_price_index':[1764,1594,1457,1493,1756,1244,1254,1175,1329,1547,1230,1375,1057,945,947,958,918,935,834,786,815,852,724,749] }
df =pd.DataFrame(Stock_Market,columns=['Year','Month','Rateofinterest','RateofUnemployment','Stock_price_index'])
X = df[['Rateofinterest','RateofUnemployment']] # here we have used 2 variables in Linear Regression using
#Multiple Variables
Y = df['Stock_price_index']
# Using sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
# prediction using sklearn
New_Rateofinterest = 3.25
New_RateofUnemployment = 7.3
print ('Predicted Stock Price Index: \n', regr.predict([[New_Rateofinterest ,New_RateofUnemployment]]))
# using statsmodels
X = sm.add_constant(X) # adding a constant
model = sm.OLS(Y, X).fit()
predictions = model.predict(X)
printmodel = model.summary()
print(printmodel)


# In[ ]:





# In[ ]:





# In[ ]:




