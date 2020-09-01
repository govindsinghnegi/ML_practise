#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd 
import mglearn
from IPython.display import display


# In[7]:


mglearn.plots.plot_linear_regression_wave()


# In[10]:


from sklearn.linear_model import LinearRegression
x,y = mglearn.dataset.make_wave(samples = 60)


# In[11]:


import mglearn
x,y = mglearn.datasets.make_wave(samples = 60)


# In[12]:


x,y = mglearn.datasets.make_wave(n_samples = 60)


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40)


# In[14]:


from sklearn.linear_model import LinearRegression
x,y = mglearn.datasets.make_wave(n_samples = 60)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40)


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x,y = mglearn.datasets.make_wave(n_samples = 60)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40)


# In[15]:


lr = LinearRegression().fit(x_train, y_train)


# In[16]:


print("lr.coeff : {}".format(lr.coef_))


# In[17]:


print("lr intercept : {}".format(lr.intercept_))


# In[25]:


print("")


# In[20]:


print(lr.score(x_train, y_train))


# In[21]:


print(lr.score(x_test, y_test))


# In[26]:


x,y = mglearn.datasets.load_extended_boston()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
lr = LinearRegression().fit(x_train, y_train)


# In[27]:


print(lr.score(x_train, y_train))


# In[28]:


print(lr.score(x_test, y_test))


# In[ ]:




