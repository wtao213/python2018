# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 20:19:13 2021

@author: wanti
"""

# https://www.datacamp.com/community/tutorials/xgboost-in-python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import datetime




# import data

df1 = pd.read_csv(r"E:\MMA\MMA869\team project\dataset\TEAM Project\0bf8bc6e-30d0-4c50-956a-603fc693d966.csv",na_values=['NA',''],encoding='utf8')
df2 = pd.read_csv(r"E:\MMA\MMA869\team project\dataset\TEAM Project\702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv",na_values=['NA',''],encoding='utf8')
df3 = pd.read_csv(r"E:\MMA\MMA869\team project\dataset\TEAM Project\4910797b-ee55-40a7-8668-10efd5c1b960.csv",na_values=['NA',''],encoding='utf8')


## join df1 and df3 it seems like df2 is testing value

df = pd.merge(df1,df3, how ='inner',on='id')
df.columns
df.info()


# general check
df['status_group'][0].strip()
df['status_group']= df['status_group'].str.strip()
df['status_group'].value_counts()



df['funder'].value_counts().head(15)

df['gps_height'].value_counts(bins=10,ascending=True,dropna=False)

df['installer'].value_counts().head(15)


df.loc[:,['status_group','funder']]

pd.crosstab(df['funder'],df['status_group'])



################################################################
# EDA


df[(df.status_group == 'functional')]

plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.hist(df.loc[(df.status_group == 'functional needs repair'),['amount_tsh']], bins=50,color = "green",ec='red',range=(0,2000),density=True,cumulative=True,alpha =0.5)

#######
# df['amount_tsh']
df['amount_tsh'].describe()
df['amount_tsh'].quantile([0.95,0.98])
df['amount_tsh'].quantile(np.arange(0.9, 1.0, 0.02))


plt.hist(df.loc[(df.status_group == 'functional'),['amount_tsh']], bins=50,color = "skyblue",ec='blue',range=(0,200),alpha =0.5)
plt.hist(df.loc[(df.status_group == 'non functional'),['amount_tsh']], bins=50,color = "red",ec='red',range=(0,2000),alpha =0.5)
plt.hist(df.loc[(df.status_group == 'functional needs repair'),['amount_tsh']], bins=50,color = "green",ec='red',range=(0,2000),alpha =0.5)
plt.xlim(0,2000)
plt.title("amount_tsh by outcome Distribution")
plt.xlabel("amount_tsh")
plt.ylabel("Count")
# add in vertial line
#plt.vlines(x=317,  colors='red', ls=':',ymin=0, ymax=25,lw=2, label=' Average = 317')
plt.legend( loc='upper right')
plt.show()

#plot version 2
sb.catplot(x="status_group", y="amount_tsh", data=df)

# create a column to binary the amount of it
df['amount_tsh_ind'] = np.where(df['amount_tsh']> 0, 1, 0)
df['amount_tsh_ind'].value_counts()



####
# df['funder']
df['funder'].describe()

df['funder'].value_counts().head(100)
df['funder'].value_counts().head(100).cumsum()


df['funder'].value_counts().head(25).plot(kind='bar')

plt.hist(df.loc[(df.status_group == 'functional'),['funder']], color = "skyblue",ec='blue',normed=True)

df.groupby(["status_group", "funder"]).size().reset_index(name="Time")

t1 = pd.crosstab(df.funder,df.status_group, normalize='columns')

# even keep top 100, remaining group still capture 25.5% of total count,and each funder min has 84 rows in
l1 = df['funder'].value_counts().head(100).index.tolist()
df['funder_v2'] = np.where(df['funder'].isin(l1), df['funder'], 'other')




t1 = pd.crosstab(df.funder_v2,df.status_group, normalize='index')

# index for function, using 0.7, base rate for cuntion is 54.3%
l1= t1[t1['functional'] > 0.7].index.tolist()
df['funder_function_ind'] = np.where(df['funder_v2'].isin(l1), 1, 0)
df['funder_function_ind'].value_counts()
# index for function, using 0.5, base rate for cuntion is 38.4%
l1= t1[t1['non functional'] > 0.5].index.tolist()
df['funder_nonfunction_ind'] = np.where(df['funder_v2'].isin(l1), 1, 0)
df['funder_nonfunction_ind'].value_counts()



##
# 'date_recorded'
df['date_recorded'].describe()

df['date_recorded'].value_counts().head(50).plot(kind='bar')
df['date_recorded'].value_counts()


df['year'] = pd.DatetimeIndex(df['date_recorded']).year
df['month'] = pd.DatetimeIndex(df['date_recorded']).month
df['weekday'] = pd.to_datetime(df['date_recorded']).dt.dayofweek















## get numeric columns

df.select_dtypes(include=np.number)


###############################################
# https://www.datacamp.com/community/tutorials/xgboost-in-python
# XGboost part from here

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

X = df.select_dtypes(include=np.number)
y = np.where(df.status_group == 'functional', 1,   #when... then
                 np.where(df.status_group == 'non functional', 0,  #when... then
                  2))
# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

data_dmatrix = xgb.DMatrix(data=X,label=y)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


xg_reg = xgb.XGBRegressor(objective ='multi:softprob', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 7, alpha = 10, n_estimators = 10,num_class=3)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))




# k-fold Cross Validation using XGBoost
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

cv_results.head()

print((cv_results["test-rmse-mean"]).tail(1))




#Visualize Boosting Trees and Feature Importance
xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

import matplotlib.pyplot as plt

xgb.plot_tree(xg_reg,num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()

xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()












#######################################################
# https://www.analyticsvidhya.com/blog/2021/08/complete-guide-on-how-to-use-lightgbm-in-python/
# LGBM

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics


# To define the input and output feature
x = df.select_dtypes(include=np.number)
y = df.status_group


# train and test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)


model = lgb.LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=42)
model.fit(x_train,y_train,eval_set=[(x_test,y_test),(x_train,y_train)],
          verbose=20,eval_metric='logloss')


print('Training accuracy {:.4f}'.format(model.score(x_train,y_train)))
print('Testing accuracy {:.4f}'.format(model.score(x_test,y_test)))


