#Import necessary packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


#Reading the data

df = pd.read_csv("D:\AIML\Dataset\ChurnData.csv")

df.head()


# In[156]:


#Selecting the columns required for the model

df = df[['tenure','age','address','income','ed','employ','equip','callcard','wireless','churn']]

df.head()


# In[4]:


#Since the DV is float datatype we will convert it to an integer

df['churn'] = df['churn'].astype('int')

df.dtypes


# In[5]:


#Checking the data to see if there are any null values 

df.info()


# In[6]:


#Declaring the IV & DV

X = np.asarray(df[['tenure','age','address','income','ed','employ','equip']]) #IV
y = np.asarray(df['churn']) #DV


# In[7]:


#Normalizing the data, since the values are in different range, we first normalize the data & split into train & test

from sklearn import preprocessing

X = preprocessing.StandardScaler().fit(X).transform(X)

X[0:2]


# In[8]:


#Now the values are transformed, we split the data into train & test

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 200)
print(f"Train set:",X_train.shape,y_train.shape)
print(f"Test set:",X_test.shape,y_test.shape)


# In[9]:


#Now we can start the modeling

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(solver='saga')
LR.fit(X_train,y_train)
LR


# In[10]:


#Using the test data to predict the output

predicted_y = LR.predict(X_test)

predicted_y


# In[11]:


#Logistic regression can also give probabilistic outputs

predicted_y_prob = LR.predict_proba(X_test)

predicted_y_prob[0:4]


# In[12]:


#Evaluating the results using F1 Score

from sklearn.metrics import f1_score

f1_score(y_test,predicted_y)


# In[13]:


#Plotting the graph by taking the 1st indexed col & 2nd last indexed col & colouring it with the actual values

X_test[:,1]
X_test[:,-2]

plt.scatter(X_test[:,1],X_test[:,-2],c=y_test)


# In[14]:


X_test[:,1]
X_test[:,-2]

plt.scatter(X_test[:,1],X_test[:,-2],c=predicted_y)

