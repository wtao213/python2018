# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import pandas.core.algorithms as algos


# import data
df = pd.read_csv(r"C:\Users\wanti\Desktop\MMA\MMA 831 Marketing Analytics\MMA831_midterm\IMB 623 VMWare- Digital Buyer Journey\IMB 623 VMWare- Digital Buyer Journey\Training.csv",na_values=['NA',9999],encoding='utf8')

df.head(10)

df.describe()

df['target'].value_counts()

list(df.columns.values)
## simple plot
plt.plot(df['tot_page_views'],df['target'],'o',color='blue')


df.iloc[:,1]
df.iloc[:,1].columns()

# fancy way to plot but very slow
sns.jointplot(x=df['tot_page_views'], y=df['tot_page_views_130d'])



###########################
#  to generate logit plot, let's create binayr target first
df['target2'] = np.where(df['target'] > 0, 1, 0)
df['target2'].value_counts()

df['target'].value_counts()




# logit plot


for i in range(1,100):
    # find the right size to bin x
    test = df.loc[:,[df.columns[i],'target2']]
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




################################
# generate a list of plot
for i in range(1,100):
    plt.figure()
    plt.plot(df.iloc[:,i],df['target'],'o',color='blue')
    plt.title(df.columns[i]+" vs target")
    plt.show()
    
    
########
# check columns with missing
 df.columns[df.isnull().any()]   
 
 
meta = df.describe()
 
meta.to_excel(r'C:\Users\wanti\Desktop\MMA\MMA 831 Marketing Analytics\MMA831_midterm\meta2.xlsx', index = False)
 

#############################################################################
# Random Forest:
#    https://www.datacamp.com/community/tutorials/random-forests-classifier-python



X = df.drop('target', 1)
X = X.select_dtypes(['number'])  #only use the numeric coloumns for now


Y = df['target']



# Import train_test_split function
from sklearn.model_selection import train_test_split


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)





















    
    