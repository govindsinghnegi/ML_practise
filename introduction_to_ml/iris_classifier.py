#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd 
import mglearn
from IPython.display import display
from sklearn.datasets import load_iris

iris_dataset = load_iris()


# In[2]:


print("keys of iris_dataset : \n{}".format(iris_dataset.keys()))


# In[3]:


print(iris_dataset['feature_names'])


# In[4]:


print(iris_dataset['target_names'])


# In[5]:


print(iris_dataset['DESCR'])


# In[6]:


print("target names : {}".format(iris_dataset['target_names']))


# In[7]:


print(iris_dataset['feature_names'])


# In[8]:


print(iris_dataset['data'].shape)


# In[9]:


print(iris_dataset['data'])


# In[10]:


print(iris_dataset['target'])


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)


# In[12]:


print(X_train.shape)


# In[13]:


print(Y_test.shape)


# In[14]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)


# In[15]:


knn.fit(X_train, Y_train)


# In[16]:


X_new = np.array([
    5, 2, 1, 0.2
])


# In[17]:


print(X_new.shape)


# In[18]:


X_new = np.arrayy(
[
    [5,2,1,0.2]
])


# In[19]:


X_new = np.array(
[
    [5,2,1,0.2]
])


# In[20]:


print(X_new.shape)


# In[21]:


prediction = knn.predict(X_new)


# In[22]:


print(prediction)


# In[23]:


y_pred = knn.predict(X_test)


# In[24]:


print(y_pred)


# In[25]:


print("accuracy : {}".format(np.mean(y_pred == y_test)))


# In[26]:


print("accuracy : {}".format(np.mean(y_pred == Y_test)))


# In[ ]:




