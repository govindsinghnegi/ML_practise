#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd 
import mglearn
from IPython.display import display


# In[8]:


x, y = mglearn.datasets.make_wave(n_samples=10)


# In[9]:


plt.plot(x, y, 'o')
plt.ylim(-3,3)
plt.xlabel("Feature")
plt.ylabel("Target")


# In[11]:


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.keys())


# In[13]:


print(cancer['feature_names'])


# In[14]:


print(cancer['target_names'])


# In[15]:


print(cancer['filename'])


# In[16]:


print(cancer['data'])


# In[17]:


print(cancer['data'].shape)


# In[18]:


print(cancer.keys())


# In[19]:


print(cancer['target'].shape)


# In[20]:


print(cancer['target'])


# In[21]:


from sklearn.datasets import load_boston
boston = load_boston()
print(boston.keys())


# In[22]:


print(boston['feature_names'])


# In[23]:


print(boston['target'])


# In[24]:


print(boston['feature_names'].shape)


# In[25]:


print(boston['data'])


# In[26]:


print(boston['data'].shape)


# In[27]:


print(boston.keys())


# In[28]:


print(boston['target'])


# In[29]:


print(boston['target'].shape)


# In[30]:


print(boston['target'])


# In[31]:


print(boston['DESC'])


# In[32]:


print(boston.keys())


# In[33]:


print(boston['DESCR'])


# In[ ]:




