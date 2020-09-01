#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd 
import mglearn
from IPython.display import display
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


# In[7]:


x,y = mglearn.datasets.load_extended_boston()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
ridge = Ridge().fit(x_train, y_train)
print("training score : {}".format(ridge.score(x_train, y_train)))
print("test score : {}".format(ridge.score(x_test, y_test)))


# In[8]:


print("with alpha = 10")
ridge = Ridge(alpha=10).fit(x_train, y_train)
print("training score : {}".format(ridge.score(x_train, y_train)))
print("test score : {}".format(ridge.score(x_test, y_test)))


# In[9]:


print("with alpha = 0.1")
ridge = Ridge(alpha=0.1).fit(x_train, y_train)
print("training score : {}".format(ridge.score(x_train, y_train)))
print("test score : {}".format(ridge.score(x_test, y_test)))


# In[ ]:




