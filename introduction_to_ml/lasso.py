#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd 
import mglearn
from IPython.display import display
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split


# In[10]:


x,y = mglearn.datasets.load_extended_boston()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
lasso = Lasso().fit(x_train, y_train)
print("with alpha = 1")
print("training score : {} ".format(lasso.score(x_train, y_train)))
print("test score : {} ".format(lasso.score(x_test, y_test)))
print("number of features used : {}".format(np.sum(lasso.coef_ != 0)))


# In[12]:


print("with alpha = 0.01")
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(x_train, y_train)
print("training score : {} ".format(lasso001.score(x_train, y_train)))
print("test score : {} ".format(lasso001.score(x_test, y_test)))
print("number of features used : {}".format(np.sum(lasso001.coef_ != 0)))


# In[14]:


print("with alpha = 0.0001")
lasso_0001 = Lasso(alpha=0.0001, max_iter=100000).fit(x_train, y_train)
print("training score : {} ".format(lasso_0001.score(x_train, y_train)))
print("test score : {} ".format(lasso_0001.score(x_test, y_test)))
print("number of features used : {}".format(np.sum(lasso_0001.coef_ != 0)))


# In[ ]:




