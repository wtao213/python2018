# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 17:58:36 2021

@author: wanti
"""



# https://www.datacamp.com/community/tutorials/xgboost-in-python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import datetime



# import data

df1 = pd.read_csv(r"E:\MMA\MMA869\team project\dataset\TEAM Project\0bf8bc6e-30d0-4c50-956a-603fc693d966.csv",na_values=['NA'],encoding='utf8')
df2 = pd.read_csv(r"E:\MMA\MMA869\team project\dataset\TEAM Project\702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv",na_values=['NA'],encoding='utf8')
df3 = pd.read_csv(r"E:\MMA\MMA869\team project\dataset\TEAM Project\4910797b-ee55-40a7-8668-10efd5c1b960.csv",na_values=['NA'],encoding='utf8')


## join df1 and df3 it seems like df2 is testing value
df = pd.merge(df1,df3, how ='inner',on='id')
df.columns
df.info()
del df1,df3


df.loc[:,df.isnull().any()].isnull().sum()

# general check
df['status_group'][0].strip()
df['status_group']= df['status_group'].str.strip()
df['status_group'].value_counts()



# create a column to binary the amount of it
df['amount_tsh_ind'] = np.where(df['amount_tsh']> 0, 1, 0)
df2['amount_tsh_ind'] = np.where(df2['amount_tsh']> 0, 1, 0)
df['amount_tsh_ind'].value_counts()




# even keep top 100, remaining group still capture 25.5% of total count,and each funder min has 84 rows in
l1 = df['funder'].value_counts().head(100).index.tolist()
df['funder_v2']  = np.where(df['funder'].isin(l1), df['funder'], 'other')
df2['funder_v2'] = np.where(df2['funder'].isin(l1), df2['funder'], 'other')



t1 = pd.crosstab(df.funder_v2,df.status_group, normalize='index')

# index for function, using 0.7, base rate for cuntion is 54.3%
l1= t1[t1['functional'] > 0.7].index.tolist()
df['funder_function_ind'] = np.where(df['funder_v2'].isin(l1), 1, 0)
df2['funder_function_ind'] = np.where(df2['funder_v2'].isin(l1), 1, 0)
df['funder_function_ind'].value_counts()
# index for function, using 0.5, base rate for cuntion is 38.4%
l1= t1[t1['non functional'] > 0.5].index.tolist()
df['funder_nonfunction_ind']  = np.where(df['funder_v2'].isin(l1), 1, 0)
df2['funder_nonfunction_ind'] = np.where(df2['funder_v2'].isin(l1), 1, 0)
df['funder_nonfunction_ind'].value_counts()





##
# 'date_recorded'
df['date_recorded'].describe()


df['year'] = pd.DatetimeIndex(df['date_recorded']).year
df['month'] = pd.DatetimeIndex(df['date_recorded']).month
df['weekday'] = pd.to_datetime(df['date_recorded']).dt.dayofweek

df2['year'] = pd.DatetimeIndex(df2['date_recorded']).year
df2['month'] = pd.DatetimeIndex(df2['date_recorded']).month
df2['weekday'] = pd.to_datetime(df2['date_recorded']).dt.dayofweek



#
df['gps_height_0_ind'] = np.where(df['gps_height']==0,1,0)
df['gps_height_high_ind'] =np.where(df['gps_height']>1600,1,0)

df2['gps_height_0_ind'] = np.where(df2['gps_height']==0,1,0)
df2['gps_height_high_ind'] =np.where(df2['gps_height']>1600,1,0)



# even keep top 100, remaining group still capture 25.5% of total count,and each funder min has 84 rows in
l1 = df['installer'].value_counts().head(100).index.tolist()
df['installer_v2'] = np.where(df['installer'].isin(l1), df['installer'], 'other')
df2['installer_v2'] = np.where(df2['installer'].isin(l1), df2['installer'], 'other')

# then look at the result by installer
t1 = pd.crosstab(df.installer_v2,df.status_group, normalize='index')

# index for function, using 0.7, base rate for cuntion is 54.3%
l1= t1[t1['functional'] > 0.7].index.tolist()
df['installer_function_ind'] = np.where(df['installer_v2'].isin(l1), 1, 0)
df2['installer_function_ind'] = np.where(df2['installer_v2'].isin(l1), 1, 0)
df['installer_function_ind'].value_counts()
# index for function, using 0.5, base rate for cuntion is 38.4%
l1= t1[t1['non functional'] > 0.5].index.tolist()
df['installer_nonfunction_ind'] = np.where(df['installer_v2'].isin(l1), 1, 0)
df2['installer_nonfunction_ind'] = np.where(df2['installer_v2'].isin(l1), 1, 0)
df['installer_nonfunction_ind'].value_counts()





# even keep top 100, remaining group still capture 80% of total count,and each funder min has 84 rows in
l1 = df['ward'].value_counts().head(100).index.tolist()
df['ward_v2'] = np.where(df['ward'].isin(l1), df['ward'], 'other')
df2['ward_v2'] = np.where(df2['ward'].isin(l1), df2['ward'], 'other')



# then look at the result by installer
t1 = pd.crosstab(df.ward_v2,df.status_group, normalize='index')
# index for function, using 0.7, base rate for cuntion is 54.3%
l1= t1[t1['functional'] > 0.7].index.tolist()
df['ward_function_ind'] = np.where(df['ward_v2'].isin(l1), 1, 0)
df2['ward_function_ind'] = np.where(df2['ward_v2'].isin(l1), 1, 0)
df['ward_function_ind'].value_counts()
# index for function, using 0.5, base rate for cuntion is 38.4%
l1= t1[t1['non functional'] > 0.5].index.tolist()
df['ward_nonfunction_ind'] = np.where(df['ward_v2'].isin(l1), 1, 0)
df2['ward_nonfunction_ind'] = np.where(df2['ward_v2'].isin(l1), 1, 0)
df['ward_nonfunction_ind'].value_counts()






###
#
l1 = df['lga'].value_counts().head(100).index.tolist()
df['lga_v2'] = np.where(df['lga'].isin(l1), df['lga'], 'other')
df2['lga_v2'] = np.where(df2['lga'].isin(l1), df2['lga'], 'other')




##
pd.qcut(df['longitude'], q=3).value_counts(dropna=False)
df['longitude_v2'] = pd.cut(df['longitude'],  bins=[-1,34,38,41],labels=["low", "medium", "high"])
df2['longitude_v2'] = pd.cut(df2['longitude'], bins=[-1,34,38,41],labels=["low", "medium", "high"])


df['population_v2'] = pd.cut(df['population'], bins=[-1,2,160,30500],labels=["low", "medium", "high"])
df2['population_v2'] = pd.cut(df2['population'], bins=[-1,2,160,30500],labels=["low", "medium", "high"])



#
df['scheme_management_v2'] = np.where(df['scheme_management'].isnull(),'Other',
         np.where(df['scheme_management']=='None','Other',df['scheme_management']))
df2['scheme_management_v2'] = np.where(df2['scheme_management'].isnull(),'Other',
         np.where(df2['scheme_management']=='None','Other',df2['scheme_management']))



#
df['extraction_type_v2'] = np.where(df['extraction_type'].isin(['other - mkulima/shinyanga','climax','walimi']),'Other',df['extraction_type'])
df2['extraction_type_v2'] = np.where(df2['extraction_type'].isin(['other - mkulima/shinyanga','climax','walimi']),'Other',df2['extraction_type'])


#
df['waterpoint_type_v2']  = np.where(df['waterpoint_type'] =='dam','cattle trough',df['waterpoint_type'])
df2['waterpoint_type_v2'] = np.where(df2['waterpoint_type'] =='dam','cattle trough',df2['waterpoint_type'])




















###########################
# Creating dummy variables:
df = pd.get_dummies(df, columns=['basin'],dummy_na=True)
df = pd.get_dummies(df, columns=['quantity'],dummy_na=True) #quantity is exactly same as quantity_group
df = pd.get_dummies(df, columns=['quality_group'],dummy_na=True) # add in this actually lower the prediction
df = pd.get_dummies(df, columns=['lga_v2'],dummy_na=True)
df = pd.get_dummies(df, columns=['waterpoint_type_v2'],dummy_na=True)
df = pd.get_dummies(df, columns=['extraction_type_v2'],dummy_na=True)
df = pd.get_dummies(df, columns=['extraction_type_class'],dummy_na=True)
df = pd.get_dummies(df, columns=['extraction_type_group'],dummy_na=True)
df = pd.get_dummies(df, columns=['source_type'],dummy_na=True)
df = pd.get_dummies(df, columns=['public_meeting'],dummy_na=True)
df = pd.get_dummies(df, columns=['permit'],dummy_na=True)
df = pd.get_dummies(df, columns=['longitude_v2'],dummy_na=True)
df = pd.get_dummies(df, columns=['management'],dummy_na=True)
df = pd.get_dummies(df, columns=['scheme_management_v2'],dummy_na=True)
df = pd.get_dummies(df, columns=['population_v2'],dummy_na=True)
df = pd.get_dummies(df, columns=['payment'],dummy_na=True)
df = pd.get_dummies(df, columns=['region'],dummy_na=True)


df2 = pd.get_dummies(df2, columns=['basin'],dummy_na=True)
df2 = pd.get_dummies(df2, columns=['quantity'],dummy_na=True)
df2 = pd.get_dummies(df2, columns=['quality_group'],dummy_na=True)
df2 = pd.get_dummies(df2, columns=['lga_v2'],dummy_na=True)
df2 = pd.get_dummies(df2, columns=['waterpoint_type_v2'],dummy_na=True)
df2 = pd.get_dummies(df2, columns=['extraction_type_v2'],dummy_na=True)
df2 = pd.get_dummies(df2, columns=['extraction_type_class'],dummy_na=True)
df2 = pd.get_dummies(df2, columns=['extraction_type_group'],dummy_na=True)
df2 = pd.get_dummies(df2, columns=['source_type'])
df2 = pd.get_dummies(df2, columns=['public_meeting'],dummy_na=True)
df2 = pd.get_dummies(df2, columns=['permit'],dummy_na=True)
df2 = pd.get_dummies(df2, columns=['longitude_v2'],dummy_na=True)
df2 = pd.get_dummies(df2, columns=['management'],dummy_na=True)
df2 = pd.get_dummies(df2, columns=['scheme_management_v2'],dummy_na=True)
df2 = pd.get_dummies(df2, columns=['population_v2'],dummy_na=True)
df2 = pd.get_dummies(df2, columns=['payment'],dummy_na=True)
df2 = pd.get_dummies(df2, columns=['region'],dummy_na=True)

# find the column in dataframe 1 not in another dataframe
df.columns.difference(df2.columns)

# drop column name
df = df.drop(labels=['source_type_nan','region_code'],axis =1)
df2 = df2.drop(labels=['region_code'],axis =1)


#######################################################
# https://www.analyticsvidhya.com/blog/2021/08/complete-guide-on-how-to-use-lightgbm-in-python/
# LGBM

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics


# To define the input and output feature
x = df.select_dtypes(include=np.number).fillna(0)
y = df.status_group
x_predict = df2.select_dtypes(include=np.number).fillna(0)

# train and test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)


model = lgb.LGBMClassifier(objective='multiclass',learning_rate=0.1,max_depth=-7,random_state=42,min_child_samples=50,
                           n_estimators=500)
model.fit(x_train,y_train,eval_set=[(x_test,y_test),(x_train,y_train)],
          verbose=50,eval_metric='logloss',early_stopping_rounds=40,)


print('Training accuracy {:.4f}'.format(model.score(x_train,y_train)))
print('Testing accuracy {:.4f}'.format(model.score(x_test,y_test)))


y_pred = model.predict(x_predict)

data_pred = pd.DataFrame()
data_pred['id'] = x_predict['id']
data_pred['status_group'] = y_pred





############################
# cross-validation
import time
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 10
strat_kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

scores = np.empty(N_SPLITS)
for idx, (train_idx, test_idx) in enumerate(strat_kf.split(x, y)):
    print("=" * 12 + f"Training fold {idx}" + 12 * "=")
    start = time.time()

    x_train, x_val = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_val = y[train_idx], y[test_idx]
    eval_set = [(x_val, y_val)]

    lgbm_clf = lgb.LGBMClassifier(objective='multiclass',learning_rate=0.1,max_depth=-7,random_state=42
                                  ,min_child_samples=50,reg_lambda=0.1,n_estimators=500)
    lgbm_clf.fit(
        x_train,
        y_train,
        eval_set=eval_set,
        verbose=50,
        early_stopping_rounds=30,
        eval_metric="logloss"
    )
    del idx,eval_set,test_idx,train_idx


print('Training accuracy {:.4f}'.format(lgbm_clf.score(x_train,y_train)))
print('Testing accuracy {:.4f}'.format(lgbm_clf.score(x_test,y_test)))


y_pred_v2 = lgbm_clf.predict(x_predict)

data_pred = pd.DataFrame()
data_pred['id'] = x_predict['id']
data_pred['status_group'] = y_pred_v2






# XGboost part from here

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

x = df.select_dtypes(include=np.number).fillna(0)
y = np.where(df.status_group == 'functional', 1,   #when... then
                 np.where(df.status_group == 'non functional', 0,  #when... then
                  2))


# data_dmatrix = xgb.DMatrix(data=x,label=y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


xg_reg = xgb.XGBRegressor(objective ='multi:softprob', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 7, alpha = 10, n_estimators = 10,num_class=3)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(x_predict)


# return the column name with max value
xgb_y =pd.DataFrame(preds).idxmax(axis=1)
xgb_y = np.where(xgb_y == 1, 'functional',   #when... then
                 np.where(xgb_y == 0, 'non functional',  #when... then
                  'functional needs repair'))



#########################
# compare all the results
result = pd.DataFrame()
result['id'] = x_predict['id']
result['lgbm'] = y_pred
result['lgbm_c'] = y_pred_v2
result['xgboost'] = xgb_y

from statistics import mode
mode(result.iloc[1,])

# add column
result['status_group'] = result.apply(lambda _: '', axis=1)
for i in range(len(result)):
    result['status_group'][i]= mode(result.iloc[i,])


data_pred= result.loc[:,['id','status_group']]

#######################
# export to csv
data_pred.to_csv(r"E:\MMA\MMA869\team project\dataset\TEAM Project\water_pump_prediction.csv",index=False)

