#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Predict which drug to use
# feature sets: Age, Sex, Blood Pressure, and Cholesterol of patients
# target set:  Drug that each patient responded to


# In[2]:


import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# In[7]:


# import data and describe
get_ipython().system('wget -O drug200.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv')

my_data = pd.read_csv("drug200.csv", delimiter=",")
my_data[0:5]
my_data.describe()
my_data.size


# In[8]:


# Declare features
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]

# Convert categorical fearures to dummies (0,1)
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]


# In[9]:


# Define target variable
y = my_data["Drug"]
y[0:5]


# In[10]:


# Train test split
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
print ('Train set:', X_trainset.shape,  y_trainset.shape)
print ('Test set:', X_testset.shape,  y_testset.shape)


# In[11]:


# define classifier and its parameters
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters
drugTree.fit(X_trainset,y_trainset)


# In[13]:


# Predict
predTree = drugTree.predict(X_testset)
#Compare prediction
print (predTree [0:5])
print (y_testset [0:5])


# In[14]:


# Evaluate model accuracy
from sklearn import metrics
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


# In[ ]:




