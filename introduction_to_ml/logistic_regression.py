#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd 
import mglearn
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[5]:


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)


# In[7]:


logistic_reg = LogisticRegression().fit(x_train, y_train)
print("with C = 1")
print("training score : {} ".format(logistic_reg.score(x_train, y_train)))
print("test score : {} ".format(logistic_reg.score(x_test, y_test)))


# In[8]:


logistic_reg = LogisticRegression(C=100).fit(x_train, y_train)
print("with C = 100")
print("training score : {} ".format(logistic_reg.score(x_train, y_train)))
print("test score : {} ".format(logistic_reg.score(x_test, y_test)))


# In[9]:


logistic_reg = LogisticRegression(C=0.01).fit(x_train, y_train)
print("with C = 0.01")
print("training score : {} ".format(logistic_reg.score(x_train, y_train)))
print("test score : {} ".format(logistic_reg.score(x_test, y_test)))


# In[10]:


logistic_reg = LogisticRegression(C=10000).fit(x_train, y_train)
print("with C = 10000")
print("training score : {} ".format(logistic_reg.score(x_train, y_train)))
print("test score : {} ".format(logistic_reg.score(x_test, y_test)))


# In[ ]:




