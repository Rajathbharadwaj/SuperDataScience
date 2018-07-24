
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('Data.csv')


# In[3]:


df.head()
df_rem = df.drop('Purchased',axis=1)


# In[4]:


X = df.iloc[:,:-1].values


# In[5]:


X


# In[6]:


y = df.iloc[:,3].values


# In[7]:


y


# In[8]:


from sklearn.preprocessing import Imputer


# In[9]:


imputer = Imputer()


# In[10]:


imputed = imputer.fit(X[:,1:3])


# In[11]:


df_rem = df.drop('Purchased',axis=1)


# In[12]:


df_rem.head(10)


# In[13]:


X[:,1:3] = imputed.transform(X[:,1:3])


# In[14]:


X[:,1:3]


# In[15]:


df


# In[16]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[17]:


label_encoder_X = LabelEncoder()


# In[18]:


X[:,0] = label_encoder_X.fit_transform(X[:,0])


# In[19]:


X


# In[20]:


ohe = OneHotEncoder(categorical_features=[0])


# In[21]:


X = ohe.fit_transform(X).toarray()


# In[22]:


X


# In[23]:


label_encoder_y = LabelEncoder()


# In[24]:


y = label_encoder_y.fit_transform(y)


# In[25]:


y


# In[28]:


from sklearn.cross_validation import train_test_split


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

