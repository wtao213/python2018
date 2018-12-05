

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 07:53:26 2018

@author: radigoz
"""


#%% import data. Original SQL code. Can be run in Python directly.

SELECT DISTINCT 
    EWC.EE_PRNT_WALLT_ID
   ,SUM(CASE WHEN CLBS.CO_ID=1 THEN CLBS.TOT_BSKT_QLFY_SL_AMT ELSE 0 END) LCLAMN
   ,SUM(CASE WHEN CLBS.CO_ID=2 THEN CLBS.TOT_BSKT_QLFY_SL_AMT ELSE 0 END) SDMAMN
   ,count(distinct CASE WHEN CLBS.CO_ID=1 THEN POS_TRANS_ID END) LCL_TXNS
   ,count(distinct CASE WHEN CLBS.CO_ID=2 THEN POS_TRANS_ID END) SDM_TXNS
    FROM RLDMPROD_V.CNSLD_EE_LGCY_IDENT_WALLT_XREF AS a
   ,RLDMPROD_V.CNSLD_LYLTY_BSKT_SUM AS CLBS
   ,(SELECT DISTINCT STR_SITE_NUM, CO_ID, RGN_NM FROM RLDMPROD_V.CNSLD_SITE_HIER 
   WHERE CO_ID=1 AND LCL_DSTRCT_NUM <>'20'
        /* Exclude Gas Bar */ and RGN_NM <>'Sales Org National' /* Exclude Joe Fresh standalone, ecommerce, Wholesale etc */
     UNION 
     SELECT DISTINCT STR_SITE_NUM, CO_ID, RGN_NM FROM RLDMPROD_V.CNSLD_SITE_HIER WHERE CO_ID=2) 
     AS All_STR
   ,EE_WALLET_CONSUMER_LATST AS EWC
   WHERE 
   CLBS.TOT_BSKT_QLFY_SL_AMT > 5
   AND a.EE_GBL_IDNT_ID = CLBS.EE_IDENTITY_ID
   AND EWC.EE_WALLT_ID = A.CONS_WALLT_ID
   AND CLBS.STR_SITE_NUM =All_STR.STR_SITE_NUM
   AND EWC.CONS_WALLT_ST_CD='ACTIVE'
   AND EWC.CONS_WALLT_STATE_CD IN ('EARNBURN','EARNONLY')
   AND CLBS.CO_ID=All_STR.CO_ID 
AND CLBS.TRANS_DT BETWEEN '2017-09-26' and '2018-09-27'
GROUP BY 1
HAVING LCL_TXNS > 0 and SDM_TXNS > 0;

#%%
import numpy as np
import pandas as pd
import os
print(os.getcwd())
#%%
df=pd.read_csv('fullpopgrthan5.csv', dtype={'EE_PRNT_WALLT_ID':'str'})
df.head()


#%%write/retrieve from saved if needed
df.to_csv('df.csv')
#%%
df=pd.read_csv('df.csv', dtype={'EE_PRNT_WALLT_ID':'str'})
df.head()
#%%
df.dtypes
#%%check dups
#%%create total Enterprise LCL & SDM transactions & sales
df['totalTXNS'] = df['LCL_TXNS'] + df['SDM_TXNS']
df['totalAMN'] = df['LCLAMN'] + df['SDMAMN']
#%%sum stats
df.describe()
#%% Remove outliers. Definitions see xls add percentile values here
criteria = 'totalTXNS > 6 and totalTXNS < 400 and totalAMN > 50 and totalAMN < 45000'
df.query(criteria, inplace=True)
#%% check values at percentiles as needed
pd.set_option('display.max_rows', 100)
df['totalTXNS'].quantile([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1,  0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 1])
df['totalAMN'].quantile([0, 0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1,  0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.9999, 1])

#%% ScatterPlot Sales and Transactions
plt.plot()
plt.figure(figsize=(10, 10))
plt.xlim([0, 450])
plt.ylim([0, 50000])
plt.title('Enterprise (LCL+SDM) Sales vs Transactions')
plt.scatter(df['totalTXNS'], df['totalAMN'])
plt.xlabel("# Transactions")
plt.ylabel("$ Sales")
plt.show()


#%%K-means
%matplotlib inline

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from numpy import random, float

data=df[['totalTXNS', 'totalAMN']]

#%% Generate 4 clusters as per the requirement. Try with more than 4 first
model = KMeans(n_clusters=4)
#%%
# I'm scaling the data to normalize it. Important for good results.#Using PCA to preprocess the data will destroy too much information that K-means needs.

model = model.fit(scale(data))
#%% Silhouette coeff. Euclidean for continuous. Gower for categorical
from sklearn.metrics import silhouette_samples, silhouette_score
silhouette_score(data, model.labels_, metric='euclidean', sample_size=50000, random_state=None)

#%% Look at the clusters each data point was assigned to
print(model.labels_)

#visualize it:
plt.figure(figsize=(15, 15))
plt.scatter(data.iloc[:,0], data.iloc[:,1], c=model.labels_.astype(float))
plt.show()
#%%add cluster labels
df['target']=model.labels_

#%% Generate summary stats by clusters
pd.options.display.float_format = "{:.2f}".format
pd.set_option('display.max_columns', 20)
df.groupby(['target']).agg({'totalAMN': [min, max, sum, "mean", "median", "count"],    # find the min, max, and sum of the duration column
                                     'totalTXNS': [min, max, sum, "mean", "median", "count"] # find the number of network type entries, or try “size” instead of count
                                     }) 
#%%joinplot running slow
g = sns.jointplot(x="totalTXNS", y="totalAMN", data=df, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$X$", "$Y$");
#%%joint KDE plot 
sns.kdeplot(df["SDM_TXNS"])
sns.kdeplot(df["LCL_TXNS"])
sns.kdeplot(df['totalTXNS'])
#%%
sns.kdeplot(df["SDM_TXNS"])
sns.kdeplot(df["LCL_TXNS"])
sns.kdeplot(df['totalTXNS'])

#%% Generate cut offs for the crosstab

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
#%%
X=data[['totalTXNS', 'totalAMN']]
Y=data[['target']]
features = ["totalTXNS", "totalAMN"] 
#%%
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
#%%
y_train['target'].value_counts(sort=False)
#%%scikit-learn uses an optimised version of the CART algorithm

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=5, min_samples_leaf=50000)
clf_gini.fit(X_train, y_train)
#%% Visualize tree
from sklearn.externals.six import StringIO 
from IPython.display import Image 
from sklearn.tree import export_graphviz
import pydotplus
from sklearn import tree

dot_data = StringIO()
export_graphviz(clf_gini, out_file=dot_data, feature_names=features, class_names=['0', '1', '2', '3'],
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
Image(graph.create_png())
#%%Test it on holdout sample
y_pred = clf_gini.predict(X_test)

from sklearn.metrics import accuracy_score

score = accuracy_score(y_test,y_pred)*100

print("Accuracy using DTree: ", round(score,1),"%")

#%%exports image
from sklearn.tree import export_graphviz
import pydotplus

dot_data = tree.export_graphviz(clf,
                                feature_names=['totalTXNS', 'totalAMN'],
                                out_file=None,
                                filled=True,
                                rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('CART.png')
graph.write_svg("CART.svg")
#%%

#%%display tree rules
#%%
def print_decision_tree(tree, feature_names=None, offset_unit='    '):
    '''Plots textual representation of rules of a decision tree
    tree: scikit-learn representation of tree
    feature_names: list of feature names. They are set to f1,f2,f3,... if not specified
    offset_unit: a string of offset of the conditional block'''

    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    value = tree.tree_.value
    if feature_names is None:
        features  = ['f%d'%i for i in tree.tree_.feature]
    else:
        features  = [feature_names[i] for i in tree.tree_.feature]        

    def recurse(left, right, threshold, features, node, depth=0):
            offset = offset_unit*depth
            if (threshold[node] != -2):
                    print(offset+"if ( " + features[node] + " <= " + str(threshold[node]) + " ) {")
                   if left[node] != -1:
                            recurse (left, right, threshold, features,left[node],depth+1)
                    print(offset+"} else {")
                    if right[node] != -1:
                            recurse (left, right, threshold, features,right[node],depth+1)
                    print(offset+"}")
            else:
                    print(offset+"return " + str(value[node]))

    recurse(left, right, threshold, features, 0,0)
    
#%%

print_decision_tree(clf_gini, data.columns)

#%%
decision_path(X, check_input=True)

#%% Once cut-offs defined generate them fr Txns and Sales
def groups(series):
    if 51<= series <1301:
        return "51-1300"
    elif 1301<= series <2226:
        return "1301-2225"
    elif 2226<= series <2611:
        return "2226-2610"
    elif 2611<= series <4071:
        return "2611-4070"
    elif 4071<= series <5301:
        return "4071-5300"
    elif 5301<= series <8941:
        return "5301-8940"
    elif 8941<= series <45000:
        return "8941-44999"    
df['Salesgroups'] = df['totalAMN'].apply(groups)
df['Salesgroups'].value_counts(sort=False)
#%%
def groups(series):
    if 7<= series <31:
        return "7-30"
    elif 31<= series <41:
        return "31-40"
    elif 41<= series <51:
        return "41-50"
    elif 51<= series <67:
        return "51-67"
    elif 67<= series <101:
        return "67-100"
    elif 101<= series <161:
        return "101-160"
    elif 161<= series <399:
        return "161-399"
df['TXNSgroups'] = df['totalTXNS'].apply(groups)
#%%
df['TXNSgroups'].value_counts(sort=True)
#%%

#%% Generate Final crosstab - counts
pd.options.display.float_format = "{:.2f}".format
pd.set_option('display.max_columns', 20)
pd.crosstab(df['TXNSgroups'],df['Salesgroups'], values=df.totalAMN, aggfunc='sum').round(0)

#%%Generate Final crosstab - % total
pd.options.display.float_format = "{:.2f}".format
pd.set_option('display.max_columns', 20)
pd.crosstab(df['TXNSgroups'],df['Salesgroups']).apply(lambda r: r/r.sum(), axis=1)
pd.crosstab(df['TXNSgroups'],df['Salesgroups']).apply(lambda r: r/len(df), axis=1)

#the END



