#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
x = np.array([
    [1,2,3],
    [4,5,6]
])
print("x: \n{}".format(x))


# In[4]:


from scipy import sparse

eye = np.eye(4)
print("Numpy array : \n {}".format(eye))


# In[5]:


sparse_matrix = sparse.csr_matrix(eye)
print("\n scipy parse csr matrix : \n {}".format(sparse_matrix))


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y, marker="x")


# In[ ]:




