# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 07:18:41 2019

finalize file for clustering

@author: lcluser
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
## import scipy import stats    not install yet

## data=pd.read_csv(r"C:\Users\lcluser\Desktop\cust_oneyr")

import pyodbc
cnxn = pyodbc.connect('Driver={SQL Server};'
			  'Server=;'
			  'Database=LCL;'
			  'UID=;'
			  'PWD=;')

## check all the customer who shopped in the passing year
## working date Nov 7th,2018
## please remove the customer who registerd less than 90?60? days
query_ty = "select CustomerID,sum(Amount) as ttl_sales_TY, sum(Qty) as ttl_qt_TY,count(distinct concat(StoreID,EndDate,TransactionID)) as ttl_txn_TY from [LCL].[dbo].[SilTransaction_TicketAmount] where CustomerID not in ('') and  EndDate between '2018-01-01 00:00:00.000' and '2018-12-31 00:00:00.000'and Amount <> 0 group by CustomerID"
df_ty=pd.read_sql(query_ty,cnxn)

##chek all customers shopped in LY
query_ly = "select CustomerID,sum(Amount) as ttl_sales_LY, sum(Qty) as ttl_qt_LY,count(distinct concat(StoreID,EndDate,TransactionID)) as ttl_txn_LY from [LCL].[dbo].[SilTransaction_TicketAmount] where CustomerID not in ('') and  EndDate between '2017-01-01 00:00:00.000' and '2017-12-31 00:00:00.000'and Amount <> 0 group by CustomerID"
df_ly=pd.read_sql(query_ly,cnxn)

## check distribution plot to remove customer
## bound $7500 , txn 150
plt.hist(df_ty['ttl_sales_TY'])
plt.hist(df_ty['ttl_sales_TY'],bins=80,range=[0,20000])


plt.hist(df_ty['ttl_txn_TY'],bins=150,range=[0,500])



## import stick-learn's cluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

X=np.array(df_ty['ttl_sales_TY'],df_ty['ttl_txn_TY'])
##




##try to redo by remove outlier first
## plot looks like no differences
## 309821 out of 310519 customers have sales <=7500, remove 698
## 307526 out of 310519 customers have txn <=150,remove 2993
test=df_ty.loc[df_ty['ttl_sales_TY']<=7500]
test=df_ty.loc[df_ty['ttl_txn_TY']<= 150]

del test

## apply the acutal filter
X2=df_ty.loc[(df_ty['ttl_sales_TY']<=7500) & (df_ty['ttl_txn_TY']>= 2)]

## try even harder filter,remaining 279901, 90% remaining
X2=df_ty.loc[(df_ty['ttl_sales_TY']<=7500) & (df_ty['ttl_sales_TY']>=20) & 
             (df_ty['ttl_txn_TY']>= 2)&(df_ty['ttl_txn_TY']<= 150)]

X2 =np.column_stack((X2['ttl_sales_TY'],X2['ttl_txn_TY']))

##warning: can only scale the data then they are float,but when ploting need convert back
kmeans=KMeans(n_clusters=4).fit(scale(X2))

centroids= kmeans.cluster_centers_
## label is each points belongs to witch groups
labels = kmeans.labels_

## ploting to see the result, method 1
plt.scatter(X2['ttl_sales_TY'],X2['ttl_txn_TY'],c=labels,s=7)

##ploting to see the result, method 2
##plt.scatter(X2.iloc[:,0],X2.iloc[:,1],c=labels,s=7)
## limit your axis
plt.xlim(0,6000)
plt.ylim(0,200)
plt.show()



## summary of clusters
X2=X2[['ttl_sales_TY','ttl_txn_TY']]
X2['target']=kmeans.labels_

#X2.groupby(['target']).agg({'ttl_sales_TY':[min,max,sum,"mean","median","count"],
#                              'ttl_txn_TY':[min,max,sum,"mean","median","count"]})


X2.groupby(['target']).agg({'ttl_sales_TY':[min,max,sum]})
X2.groupby(['target']).agg({'ttl_sales_TY':["mean","median","count"]})

X2.groupby(['target']).agg({'ttl_txn_TY':[min,max,sum]})
X2.groupby(['target']).agg({'ttl_txn_TY':["mean","median","count"]})


## cross tab
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


X=X2[['ttl_sales_TY','ttl_txn_TY']]
Y=X2[['target']]
features=['ttl_sales_TY','ttl_txn_TY']

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=100)

y_train['target'].value_counts(sort=False)

## min_smples_leaf is the mini samples number in each leaf node, cant be too large
## after testing, choose depth 5, and min sample leasf =500
clf_gini=DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=5,
                                min_samples_leaf=2000)

clf_gini.fit(X_train,y_train)


## Jan,14th, 2019 function using testing the accuracy
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
        

valiplot(10,1000)
valiplot(9,2000)
valiplot(9,500)


##visulization of the tree
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
from sklearn import tree


##check path
from os import system
dot_data = open("C:/Users/lcluser/Desktop/dtree2.dot",'w')
tree.export_graphviz(clf_gini,out_file = dot_data,feature_names = features,
                     class_names=['0','1','2','3'],
                filled=True, rounded =True, special_characters=True)
dot_data.close()
system("dot -Tpng C:/Users/lcluser/Desktop/dtree2.dot -o C:/Users/lcluser/Desktop/dtree2.png")



## check the accuracy
y_pred =clf_gini.predict(X_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,y_pred)*100

print("Accuracy using DTree:",round(score,1),"%")


##export tree

dot_data= tree.export_graphviz(clf_gini,
                               out_file = None,
                               feature_names = features,
                filled=True, rounded =True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)

graph.write_png('CART.png')

## jan 10 ,2019
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



######################
## try sample online to find the cutoff value
##

def tree_to_code(tree,feature_names):
    tree_=tree.tree_
    
    if feature_names is None:
        features =['f%d'%i for i in tree.tree_.feature]
    else:
        features =[feature_names[i] for i in tree.tree_.feature]
    ##print "def tree({}):".format(", ".join(feature_names))
    
    def recurse(node,depth):
        indent= " "*depth
        if tree_.feature[node] != -2 :
            name= features[node]
            threshold= tree_.threshold[node]
            print ("{} if {} <= {} :".format(indent,name,threshold))
            recurse(tree_.children_left[node],depth+1)
            print ("{} else: # if {} > {}".format(indent,name,threshold))
            recurse(tree_.children_right[node],depth+1) 
        else:
            print ("{} return {}".format(indent,tree_.value[node]))
        
    recurse(0,1)
    
tree_to_code(clf_gini,X.columns)


