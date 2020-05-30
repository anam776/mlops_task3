#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


# In[6]:


dataSet = pd.read_csv('Downloads/SalaryData.csv')


# In[7]:


dataSet


# In[8]:


dataSet.head(5)


# In[10]:


y = dataSet['Salary']


# In[11]:


y


# In[15]:


dataSet .columns


# In[19]:


x = dataSet['YearsExperience']


# In[20]:


x


# In[21]:



X =  x.values


# In[22]:


X


# In[25]:


import cv2


# In[27]:


p = X.reshape(30, 1)


# In[28]:


p 


# In[40]:


from sklearn .linear_model import LinearRegression


# 

# In[42]:


mind = LinearRegression ()


# In[43]:


mind.fit (p,y)


# In[48]:


mind.predict([[20]])


# mind.coef_

# In[49]:


mind.coef_


# In[50]:


mind.intercept_


# In[51]:


9449.96232146* 20 +25792.20019866871


# In[52]:


from sklearn . externals import joblib


# In[53]:


joblib.dump(mind,'salarymodel.pkg')


# In[54]:


model = joblib.load('salarymodel.pkg')


# In[57]:


model.predict([[10]])


# In[ ]:




