#!/usr/bin/env python
# coding: utf-8

# ### Import library

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ### Load data

# In[2]:


df_credit_card = pd.read_csv(r"E:\Users\User\Documents\python\bankcraditcard.csv")


# In[3]:


df_credit_card.head()


# In[4]:


df_credit_card.columns


# In[5]:


df_credit_card.info()


# In[37]:


df_credit_card.describe()


# In[6]:


df_credit_card['Default_Payment'].describe()


# In[35]:


df_credit_card['Academic_Qualification'].describe()


# In[36]:


df_credit_card['Marital'].describe()


# Remove "Customer ID" since all the values are unique and does not contribute any information for analysis

# In[7]:


#Drop customer id column
df_credit_card.drop('Customer ID', axis=1,inplace=True);


# ### Correlation matrix 

# In[8]:


#calculating correlation among numeric variable 
corr_matrix = df_credit_card.corr() 

#plot correlation matrix
plt.figure(figsize=(20,12))
sns.heatmap(corr_matrix,
            cmap='coolwarm',
            annot=True);


# The above result shows all independent variables are slightly correlated with target variable. But independent variables have multicolinearity. For example: "March_Bill_Amount" is highly correlated with "April_Bill_Amount".
# 
# We have to consider all the variables to build a machine learning algorithm that decides whether a customer will default in the next month or not.

# ### Splitting the dataset into input and output 

# In[9]:


X = df_credit_card.drop('Default_Payment',axis=1)
y = df_credit_card.loc[:,'Default_Payment']


# ## Implementing logistic regression with SGD 

# ### Splitting input data into training dataset and testing dataset 

# In[10]:


#import train and test split module from sklearn
from sklearn.model_selection import train_test_split

#split train and test datset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


# ### Create the logistic regression model with SGD - with some basic parameters 

# In[11]:


from sklearn.linear_model import SGDClassifier
logreg_SGD = SGDClassifier(loss="log",max_iter=1000, early_stopping=True)


# ### Training the model 

# In[12]:


logreg_SGD.fit(X_train,y_train)


# In[13]:


#### Model Score - Return the mean accuracy on the given test data and labels.


# In[14]:


logreg_SGD.score(X_train,y_train)


# ### Predicting the test set results and caculating the accuracy 

# In[15]:


pred_test = logreg_SGD.predict(X_test)


# In[16]:


print('Accuracy of logistic regression classifier on test set: {:.4f}'.format
      (logreg_SGD.score(X_test, y_test)))


# ### Checking other parameters for model 

# In[17]:


#import confusion matrix from sklearn
from sklearn.metrics import confusion_matrix
#create confusion matrix table
confusion_matrix = confusion_matrix(y_test, pred_test)
print(confusion_matrix)


# In[18]:


#import classification report from sklearn
from sklearn.metrics import classification_report
print(classification_report(y_test, pred_test))


# ### Cross validation - evaluating estimator performance 

# In[19]:


from sklearn import model_selection
#import cross validation score model from sklearn
from sklearn.model_selection import cross_val_score


#creat a logistic regression model with SGD
modelCV = SGDClassifier(loss="log", tol=0.01,eta0=1.0,learning_rate="adaptive", max_iter=1000, early_stopping=True)

#call cross_val_score
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=10 , scoring='accuracy')
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
print(results)


# ### Training the model with grid search 

# In[20]:


from sklearn.linear_model import SGDClassifier
from time import time
from sklearn.model_selection import GridSearchCV

logreg_SGD = SGDClassifier()


# In[21]:


params = {
    "loss" : ["hinge", "log", "squared_hinge", "modified_huber"],
    "alpha" : [0.0001, 0.001, 0.01, 0.1],
    "penalty" : ["l2", "l1", "none"],
    "max_iter": [10,100,1000]
}


# ### Create a model with grid search 

# In[22]:


# ignore the deprecation warning
warnings.filterwarnings("ignore")
clf = GridSearchCV(logreg_SGD, param_grid=params)


# In[23]:


clf.fit(X_train,y_train)


# ### Print score and best parameter values 

# In[24]:


# View the accuracy score
print('Best score for data1:', clf.best_score_)


# In[25]:


clf.best_estimator_


# In[26]:


# View the best parameters for the model found using grid search
print('Best C:',clf.best_estimator_.C) 
print('Best alpha:',clf.best_estimator_.alpha) 
print('Best max_iter:',clf.best_estimator_.max_iter)
print('Best tol:',clf.best_estimator_.tol) 
print('Best eta0:',clf.best_estimator_.eta0) 
print('Best learning rate:',clf.best_estimator_.learning_rate) 
print('Best loss:',clf.best_estimator_.loss)


# ### Build a model using best parameters

# create a logistic regression classifier with sgd

# In[27]:


New_Model = SGDClassifier(alpha=0.001, average=False, class_weight=None,
              early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
              l1_ratio=0.15, learning_rate='optimal', loss='log',
              max_iter=100, n_iter_no_change=5, n_jobs=None, penalty='l1',
              power_t=0.5, random_state=None, shuffle=True, tol=0.001,
              validation_fraction=0.1, verbose=0, warm_start=False)


# ### Training the model 

# In[28]:


# ignore the deprecation warning
warnings.filterwarnings("ignore")

New_Model.fit(X_train,y_train)


# In[29]:


New_Model.score(X_train,y_train)


# ### Predictions on test data 

# In[30]:


pred_test = New_Model.predict(X_test)


# In[31]:


print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(New_Model.score(X_test, y_test)))


# ### Confusion matrix 

# In[32]:


#import confusion matrix from sklearn
from sklearn.metrics import confusion_matrix
#create confusion matrix table
confusion_matrix = confusion_matrix(y_test, pred_test)
print(confusion_matrix)


# ### Classification report 

# In[33]:


#import classification report from sklearn
from sklearn.metrics import classification_report
print(classification_report(y_test, pred_test))


# ## ROC curve from sklearn import metrics 

# In[34]:


#import metrics from sklearn to calculate auc score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

#calculate auc score
logit_roc_auc = roc_auc_score(y_test, New_Model.predict(X_test))

#Prediction on test data based on the number of thresholds and calculate the false positive rate and true positive rate.
fpr, tpr, thresholds = roc_curve(y_test, New_Model.predict_proba(X_test)[:,1])

# create a figure object
plt.figure()

#plot false positive rate value and true positive rate value and area under curve value
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

#dignal dotted red line
plt.plot([0, 1], [0, 1],'r--')

#x-axis limitation
plt.xlim([0.0, 1.0])

#y-axis limitaion
plt.ylim([0.0, 1.05])

#x-axis label
plt.xlabel('False Positive Rate')

#y-axis label
plt.ylabel('True Positive Rate')

#title for plot
plt.title('Receiver operating characteristic')

#print legend on lower right
plt.legend(loc="lower right")

#save the plot as a image
plt.savefig('Log_ROC')
#print the plot
plt.show()


# AUC ranges from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0.

# ### Conclusion 

# The logistic regression with SGD classifier has an accuracy of 78% when predicting the default of the the credit card holders. However the Recall is 0 which clearly incdicate the model's inefficiency to predict the defaulters
