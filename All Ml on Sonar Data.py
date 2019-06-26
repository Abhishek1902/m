#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


S=pd.read_csv("Sonar.csv")


# In[3]:


S.head(2)


# In[4]:


dum=pd.get_dummies(S,drop_first=True)


# In[6]:


X=dum.iloc[:,0:60]
y=dum.iloc[:,-1]


# In[8]:


#Importing Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[9]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=25666)


# In[10]:


#navie bays
from sklearn.naive_bayes import GaussianNB


# In[11]:


gaussian=GaussianNB()
gaussian.fit(X_train,y_train)
ypred=gaussian.predict(X_test)


# In[12]:


print(accuracy_score(y_test,ypred))
print(confusion_matrix(y_test,ypred))
print(classification_report(y_test,ypred))


# In[13]:


#roc curve
from sklearn.metrics import roc_auc_score,roc_curve


# In[15]:


# Compute predicted probabilities: y_pred_prob
y_pred_prob = gaussian.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)


# In[17]:


# Plot ROC curve
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
roc_auc_score(y_test, y_pred_prob)


# # Logistic Regression

# In[18]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


# logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)


# In[20]:


# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))


# In[21]:


# Import necessary modules
from sklearn.metrics import roc_curve, roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)


# In[22]:


# Plot ROC curve
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
roc_auc_score(y_test, y_pred_prob)


# # Random Forest

# In[23]:


from sklearn.ensemble import RandomForestClassifier


# In[24]:


rf=RandomForestClassifier(random_state=14512)
rf.fit(X_train,y_train)
ypred=rf.predict(X_test)


# In[25]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[26]:


print(mean_squared_error(y_test,ypred))
print(mean_absolute_error(y_test,ypred))
print(r2_score(y_test,ypred))


# # Random Forest Regressor 

# In[27]:


from sklearn.ensemble import RandomForestRegressor


# In[28]:


rfr=RandomForestRegressor()
rfr.fit(X_train,y_train)
ypred=rfr.predict(X_train)


# In[29]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[35]:


#print(mean_absolute_error(y_test,ypred))
#print(mean_squared_error(y_test,ypred))
#print(r2_score(y_test,ypred))


# # SVR Grid Search CV

# In[37]:


from sklearn.model_selection import GridSearchCV
import numpy as np

from sklearn.svm import SVR

C_range = np.array([0.01,0.05,0.1,1,1.5,1.7,2,4])

param_grid = dict(C=C_range)


# In[38]:


svr = SVR(kernel='linear')

svmGrid = GridSearchCV(svr, param_grid=param_grid, cv=5,scoring='neg_mean_absolute_error')

svmGrid.fit(X, y)


# In[39]:


# Best Parameters
print(svmGrid.best_params_)


# In[40]:


print(svmGrid.best_score_)


# # Support Vector Machine

# In[42]:


from sklearn.svm import SVC

svc = SVC(probability = True,kernel='linear')
fitSVC = svc.fit(X_train, y_train)
y_pred = fitSVC.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred))


# # Decision Tree Classifier and regressior 

# In[43]:


from sklearn.tree import DecisionTreeClassifier


# In[48]:


clf=DecisionTreeClassifier(random_state=25415)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)


# In[49]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))


# # Decision Tree Regressior

# In[50]:


from sklearn.tree import DecisionTreeRegressor


# In[51]:


clf = DecisionTreeRegressor(random_state=2019)
clf2 = clf.fit(X_train, y_train)
y_pred = clf2.predict(X_test)


# In[52]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test, y_pred) ** 0.5)
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))


# In[53]:


depth_range = [3,4,5,6,7,8,9]
minsplit_range = [5,10,20,25,30]
minleaf_range = [5,10,15]

parameters = dict(max_depth=depth_range,
                  min_samples_split=minsplit_range, 
                  min_samples_leaf=minleaf_range)

from sklearn.model_selection import GridSearchCV
clf = DecisionTreeRegressor(random_state=2018)
cv = GridSearchCV(clf, param_grid=parameters,
                  cv=5,scoring='neg_mean_absolute_error')

cv.fit(X,y)
# Best Parameters
print(cv.best_params_)

print((-1)*cv.best_score_)

cv.best_estimator_


# # XGBOOST Algorithm CLASSifier

# In[55]:


from sklearn.ensemble import GradientBoostingClassifier


# In[56]:


clf = GradientBoostingClassifier(random_state=1200)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)


# In[57]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))


# # XGBoost Regression

# In[58]:


from sklearn.ensemble import GradientBoostingRegressor


# In[59]:


clf = GradientBoostingRegressor(random_state=1200)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)


# In[60]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

