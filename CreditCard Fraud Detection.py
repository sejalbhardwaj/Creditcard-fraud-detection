#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Packages related to general operating system & warnings
import os 
import warnings
warnings.filterwarnings('ignore')
#Packages related to data importing, manipulation, exploratory data #analysis, data understanding
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
#Packages related to data visualizaiton
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#Setting plot sizes and type of plot
plt.rc("font", size=14)
plt.rcParams['axes.grid'] = True
plt.figure(figsize=(6,3))
plt.gray()
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.preprocessing import  PolynomialFeatures, KBinsDiscretizer, FunctionTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, OrdinalEncoder
import statsmodels.formula.api as smf
import statsmodels.tsa as tsa
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import BaggingClassifier, BaggingRegressor,RandomForestClassifier,RandomForestRegressor 
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[2]:


data=pd.read_csv("creditcard.csv")


# In[3]:


data.head()


# In[4]:


Total_transactions = len(data)
normal = len(data[data.Class == 0])
fraudulent = len(data[data.Class == 1])
fraud_percentage = round(fraudulent/normal*100, 2)
print('Total number of Trnsactions are {}'.format(Total_transactions))
print('Number of Normal Transactions are {}'.format(normal))
print('Number of fraudulent Transactions are {}'.format(fraudulent))
print('Percentage of fraud Transactions is {}'.format(fraud_percentage))


# In[5]:


data.info()


# In[6]:


sc = StandardScaler()
amount = data['Amount'].values
data['Amount'] = sc.fit_transform(amount.reshape(-1, 1))


# In[7]:


data.drop(['Time'], axis=1, inplace=True)


# In[8]:


data.drop_duplicates(inplace=True)


# In[9]:


X = data.drop('Class', axis = 1).values
y = data['Class'].values


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)


# In[11]:


DT = DecisionTreeClassifier(max_depth = 4, criterion = 'entropy')
DT.fit(X_train, y_train)
dt_yhat = DT.predict(X_test)


# In[12]:


print('Accuracy score of the Decision Tree model is {}'.format(accuracy_score(y_test, dt_yhat)))


# In[13]:


print('F1 score of the Decision Tree model is {}'.format(f1_score(y_test, dt_yhat)))


# In[14]:


confusion_matrix(y_test, dt_yhat, labels = [0, 1])


# In[27]:


n = 7
KNN = KNeighborsClassifier(n_neighbors = n)
KNN.fit(X_train, y_train)
knn_yhat = KNN.predict(X_test)


# In[16]:


print('Accuracy score of the K-Nearest Neighbors model is {}'.format(accuracy_score(y_test, knn_yhat)))


# In[17]:


print('F1 score of the K-Nearest Neighbors model is {}'.format(f1_score(y_test, knn_yhat)))


# In[18]:


lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_yhat = lr.predict(X_test)


# In[19]:


print('Accuracy score of the Logistic Regression model is {}'.format(accuracy_score(y_test, lr_yhat)))


# In[20]:


print('F1 score of the Logistic Regression model is {}'.format(f1_score(y_test, lr_yhat)))


# In[21]:


svm = SVC()
svm.fit(X_train, y_train)
svm_yhat = svm.predict(X_test)


# In[22]:


print('Accuracy score of the Support Vector Machines model is {}'.format(accuracy_score(y_test, svm_yhat)))


# In[23]:


print('F1 score of the Support Vector Machines model is {}'.format(f1_score(y_test, svm_yhat)))


# In[24]:


rf = RandomForestClassifier(max_depth = 4)
rf.fit(X_train, y_train)
rf_yhat = rf.predict(X_test)


# In[25]:


print('Accuracy score of the Random Forest model is {}'.format(accuracy_score(y_test, rf_yhat)))


# In[26]:


print('F1 score of the Random Forest model is {}'.format(f1_score(y_test, rf_yhat)))


# In[ ]:




