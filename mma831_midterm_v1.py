# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 19:28:34 2021

@author: wanti
"""
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import pandas.core.algorithms as algos
import collections
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score



# import data
df = pd.read_csv(r"C:\Users\wanti\Desktop\MMA\MMA 831 Marketing Analytics\MMA831_midterm\IMB 623 VMWare- Digital Buyer Journey\IMB 623 VMWare- Digital Buyer Journey\Training.csv",na_values=['NA',9999],encoding='utf8')

df.describe()

# check target base rate
df['target'].value_counts()

# create binary target
df['target2'] = np.where(df['target'] > 0, 1, 0)
df['target2'].value_counts()


########
# check columns with missing
df.columns[df.isnull().any()]   

# get metadata and export
meta = df.describe()
 
meta.to_excel(r'C:\Users\wanti\Desktop\MMA\MMA 831 Marketing Analytics\MMA831_midterm\meta2.xlsx', index = False)
 


###########################
#  generate logit plot
# logit plot


for i in range(1,100):
    # find the right size to bin x
    test = df.loc[:,[df.columns[i],'target2']]
    test = test.dropna()
    bins = np.unique(algos.quantile(test.loc[:,df.columns[i]], np.linspace(0, 1, 11)))
    test['bin'] = pd.cut(test.loc[:,df.columns[i]], bins ,right=False)
    
    # using bin to get logit y
    tt = test.groupby(['bin'],as_index= False).agg({ 'target2':['count','sum'], 
                         df.columns[i]:'mean'})
    tt['logity'] = np.log((tt.iloc[:,2] + 1)/(tt.iloc[:,1] -tt.iloc[:,2] + 1))
    
    # plot it
    plt.figure()
    plt.plot(tt.iloc[:,3],tt['logity'],color='blue')
    plt.title(df.columns[i]+" vs target")
    plt.show()








# we only want to check numeric columns for now, get the variables with more than one bin
df2  = df.select_dtypes(['number'])
len2 = len(df2.columns)

x_list = []
for i in range(1,len2):
    # find the right size to bin x
    test = df2.loc[:,[df2.columns[i],'target2']]
    test = test.dropna()
    bins = np.unique(algos.quantile(test.loc[:,df2.columns[i]], np.linspace(0, 1, 11)))
    if len(bins) > 1:
        x_list += [df2.columns[i]]
    
    
del df2,len2   
    
## check some no plot variable
# survey_display_events only 3 1 remaining all 0, so only one bine [0,1] no plot can draw
df['survey_display_events'].describe()
df['survey_display_events'].value_counts()
np.unique(algos.quantile(df['survey_display_events'], np.linspace(0, 1, 11)))






#############################################################
# categorical dummy coding
# One-hot encode the data using pandas get_dummies
features = df.drop(['target','target2'], 1)
features = features.drop(df.columns[df.isnull().any()], 1)
#one coding
features = pd.get_dummies(features)


# modeling start here
X = features
Y = df['target']

collections.Counter(Y)


# Binarize the output
Y = label_binarize(Y, classes=[0, 1, 2,3,4,5])
n_classes = Y.shape[1]



# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state= 888)


#Create a Gaussian Classifier
# n_estimators :The number of trees in the forest.
# min_samples_split: The minimum number of samples required to split an internal node:
clf=RandomForestClassifier(n_estimators=100, max_depth= 8,min_samples_split=200)


#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred= clf.predict(X_test)




# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    




#######################################################################
# if we only use binary target

#X = df.drop('target', 1)
#X = X.select_dtypes(['number'])  #only use the numeric coloumns for now
X = features
Y = df['target2']
    
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state= 888)
 
#Create a Gaussian Classifier
# n_estimators=50,  max_depth= 8,min_samples_split=200,  roc = 0.6664613138019942
# n_estimators=100, max_depth= 8,min_samples_split=200,  roc = 0.8332306569009971
# n_estimators=150, max_depth= 8,min_samples_split=200, roc = 0.9146213678103201
# n_estimators=200, max_depth= 8,min_samples_split=200, roc = 0.8458506267790935
# n_estimators=250, max_depth= 8,min_samples_split=200, roc = 0.7937220023837653

# n_estimators=150, max_depth= 7,min_samples_split=200, roc = 0.8357409606856382
# n_estimators=150, max_depth= 9,min_samples_split=200, roc = 0.8853935067737705

# n_estimators=150, max_depth= 8,min_samples_split=150, roc = 0.8980477021293123
# n_estimators=150, max_depth= 8,min_samples_split=250, roc = 0.8116363836507071
clf=RandomForestClassifier(n_estimators=150, max_depth= 8,min_samples_split=250)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred= clf.predict(X_test)
y_prob= clf.predict_proba(X_test)

collections.Counter(y_pred)


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)




























