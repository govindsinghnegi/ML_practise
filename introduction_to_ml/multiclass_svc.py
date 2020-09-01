#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd 
import mglearn
from IPython.display import display
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


# In[3]:


from sklearn.datasets import make_blobs
x,y = make_blobs(random_state=42)
linear_svm = LinearSVC().fit(x,y)
print("coefficient shape ", linear_svm.coef_.shape)
print("intercept shape ", linear_svm.intercept_.shape)


# print(linear_svm.coef_.shape)

# In[4]:


print(linear_svm.coef_)


# In[ ]:


# that's it

