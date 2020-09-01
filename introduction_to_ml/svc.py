#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd 
import mglearn
from IPython.display import display
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


# In[3]:


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)


# In[4]:


svc = LinearSVC().fit(x_train, y_train)
print("with C = 1")
print("training score : {} ".format(svc.score(x_train, y_train)))
print("test score : {} ".format(svc.score(x_test, y_test)))


# In[5]:


svc = LinearSVC(C=100).fit(x_train, y_train)
print("with C = 100")
print("training score : {} ".format(svc.score(x_train, y_train)))
print("test score : {} ".format(svc.score(x_test, y_test)))


# In[6]:


svc = LinearSVC(C=0.01).fit(x_train, y_train)
print("with C = 0.01")
print("training score : {} ".format(svc.score(x_train, y_train)))
print("test score : {} ".format(svc.score(x_test, y_test)))


# In[7]:


svc = LinearSVC(C=10000).fit(x_train, y_train)
print("with C = 10000")
print("training score : {} ".format(svc.score(x_train, y_train)))
print("test score : {} ".format(svc.score(x_test, y_test)))


# In[ ]:




