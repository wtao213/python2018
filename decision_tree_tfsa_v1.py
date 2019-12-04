# -*- coding: utf-8 -*-
"""
Created on Mon Dec 2 11:36:23 2019

@author: 012790
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




## import prepared data
df = pd.read_csv(r"C:\Users\012790\Desktop\TFSA\tfsa_cross_sell76k.csv",encoding='windows-1252')


## filter out the results is null
## df2 =df[df.EquityCAD_dec18.isnull()]

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


## define your x and y
## remove missing values
df2 = df[['MTjan19','RRSP_ind','join_month','Margin_ind','age_jan19','EquityCAD_jan19','rev_ly','exp_ly','AssetIn_ttl_lq','target']]
df3= df2.dropna()
del df2

## dealing with inbalance data
t1=df3[df3.target == 0]
t2=df3[df3.target == 1]

t3 = t1.sample(frac=0.05, random_state=1) ## get sample of target =0
df3=pd.concat([t2,t3],ignore_index=True) ## bomvine these two dataframe together
del t1,t2,t3



#######################

X= df3[['MTjan19','RRSP_ind','join_month','Margin_ind','age_jan19','EquityCAD_jan19','rev_ly','exp_ly','AssetIn_ttl_lq']]
Y= df3[['target']]

features=['MTjan19','RRSP_ind','join_month','Margin_ind','age_jan19','EquityCAD_jan19','rev_ly','exp_ly','AssetIn_ttl_lq']


##
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=100)
y_train['target'].value_counts(sort=False)



## based on different depth, and mini sample size
def valiplot(depth,sample):
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=100)
    for i in range (2,depth):
        
        clf_gini=DecisionTreeClassifier(criterion="gini",random_state=100,
                                        max_depth=i,min_samples_leaf=sample) 
        clf_gini.fit(X_train,y_train)
        y_pred =clf_gini.predict(X_test)
        score = accuracy_score(y_test,y_pred)*100

        print("depth ",i," Accuracy using DTree:",round(score,1),"%")
        
# overfitting check, depth =50 and 
valiplot(10,50)



## min_smples_leaf is the mini samples number in each leaf node, cant be too large
clf_gini=DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=5,
                                min_samples_leaf=50)

clf_gini.fit(X_train,y_train)




##visulization of the tree
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
from sklearn import tree
import graphviz


##check path ## try to solve the path issue
## don't have the access to change path, use online graphiz to translate the graph
## https://dreampuf.github.io/GraphvizOnline/ or search graphvia online, so copy past your dot file
from os import system

dot_data = open("C:/Users/012790/Desktop/dtree2.dot",'w')
tree.export_graphviz(clf_gini,out_file = dot_data,feature_names = features,
                     class_names=['a','b','no invite'],
                filled=True, rounded =True, special_characters=True)
dot_data.close()
system("dot -Tpng C:/Users/012790/Desktop/dtree2.dot -o C:/Users/012790/Desktop/dtree2.png")



## check the accuracy
## clf_gini.predict this predict the class they belongs to 
y_pred =clf_gini.predict(X_test)

## this return the predict probablity
##probs = clf_gini.predict_proba(X_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,y_pred)*100

print("Accuracy using DTree:",round(score,1),"%")








############################################################
fpr, tpr, thresholds = roc_curve(y_test,y_pred)


## figure out roc curve
# Compute ROC curve and ROC area for each class


from itertools import cycle


from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


fpr, tpr, thresholds = roc_curve(y_test,y_pred)

## calculate auc: reveiver Operating Characteristic Curve
auc = roc_auc_score(y_test,y_pred)

print('AUC: %.3f' % auc)













fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3183):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])




## Plot of a ROC curve for a specific class
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()









fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
##########################################################################


###############################################################################################



## look at the cutting values
## display tree rules
## macro function for decision tree rule print
def print_decision_tree(tree,feature_names=None,offset_unit=''):
    
    left      =tree.tree_.children_left
    right     =tree.tree_.children_right
    threshold =tree.tree_.threshold
    value = tree.tree_.value
    if feature_names is None:
        features =['f%d'%i for i in tree.tree_.feature]
    else:
        features =[feature_names[i] for i in tree.tree_.feature]
        
    def recurse(left,right,threshold,features,node,depth=0):
        offset = offset_unit*depth
        if (threshold[node] != -2):
                print (offset + "if(" + str(features[node]) + "<=" + str(threshold[node])+"){")                                                                                                                                      
                if left[node] != -1 :
                    recurse (left,right,threshold,features,left[node],depth+1)
                print (offset + "} else {")
                if right[node] != -1 :
                    recurse(left,right,threshold,features,right[node],depth+1)
                    print(offset +"}")
        else:
                print(offset + "return" + str(value[node]))
    recurse(left,right,threshold,features,0,0)
    
    
print_decision_tree(clf_gini,X.columns)


#### visual decision tree

dot_data= tree.export_graphviz(clf_gini,
                               out_file = None,
                               feature_names = features,
                filled=True, rounded =True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)

graph.write_png('CART.png')

Image(graph.create_png())




############################################################################
##      https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
## ROC curve












########################################################################
## complete code for logistic regression
##


# roc curve and auc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot


# generate 2 class dataset
# X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
# split into train/test sets
# trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
df2 = df[['MTjan19','RRSP_ind','join_month','Margin_ind','age_jan19','EquityCAD_jan19','rev_ly','exp_ly','AssetIn_ttl_lq','target']]
df3= df2.dropna()

## transfer equity to x^2 as plot is non-linear

X= df3[['MTjan19','RRSP_ind','join_month','Margin_ind','age_jan19','EquityCAD_jan19','rev_ly','exp_ly','AssetIn_ttl_lq']]
Y= df3[['target']]

trainX, testX, trainy, testy=train_test_split(X,Y,test_size=0.5,random_state=100)
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(testy))]
# fit a model
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
# predict probabilities
lr_probs = model.predict_proba(testX)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(testy, ns_probs)
lr_auc = roc_auc_score(testy, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()







#########################################################################
## look at roc by decison tree
##

ns_probs = [0 for _ in range(len(y_test))]

y_pred = clf_gini.predict_proba(X_test)
lr_probs = y_pred
lr_probs =lr_probs[:,1]


ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)


# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision tree: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Selection')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Decision Tree')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()