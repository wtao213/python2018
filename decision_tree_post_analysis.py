# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:52:24 2019

@author: 012790
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

## read your data
## data = pd.read_csv(r"C:\Users\012790\Desktop\post_analytics\act_info_47k_v1.csv")
## adding encoding info will solve the problem....eventhough don't know what happend
# del data

df = pd.read_csv(r"C:\Users\012790\Desktop\post_analytics\client_master_ml.csv",encoding='windows-1252')




## import stick-learn's cluster
## this part is using for clustering, don't really need right now, as we already have build in 3 groups
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale



#############################################################################
## unbalanced data, since onboarding group is too small, want to repeat that group 10 times
##
df1=df[df.Type =='Onboarding Team']
dfr = pd.concat([df1]*10, ignore_index=True)

df2=df[df.Type !='Onboarding Team']

## remanipulate version is 14k act
df=pd.concat([dfr,df2],ignore_index=True)

## cross tab
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

## all number variables
# X= df[['ttl_asset_3mt','equity_3mt_amt_sum','join_age','MTD_1']]

## only use the attribute info and then compare to see the improvement
X= df[['join_age','MTD_1','class_ind_n','FirstAccountType_a_n']]
Y= df[['Type']]
features=['join_age','MTD_1','class_ind_n','FirstAccountType_a_n']

## add in with new info
X= df[['ttl_asset_3mt','equity_3mt_amt_sum','join_age','MTD_1','open_act','class_ind_n','FirstAccountType_a_n']]
Y= df[['Type']]
features=['ttl_asset_3mt','equity_3mt_amt_sum','join_age','MTD_1','open_act','class_ind_n','FirstAccountType_a_n']

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=100)
y_train['Type'].value_counts(sort=False)


## min_smples_leaf is the mini samples number in each leaf node, cant be too large
clf_gini=DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=5,
                                min_samples_leaf=50)


clf_gini.fit(X_train,y_train)


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
y_pred =clf_gini.predict(X_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,y_pred)*100

print("Accuracy using DTree:",round(score,1),"%")



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










