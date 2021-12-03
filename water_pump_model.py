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
import re
import math
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# import data

df1 = pd.read_csv(r"E:\MMA\MMA869\team project\dataset\TEAM Project\0bf8bc6e-30d0-4c50-956a-603fc693d966.csv",na_values=['NA'],encoding='utf8')
df2 = pd.read_csv(r"E:\MMA\MMA869\team project\dataset\TEAM Project\702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv",na_values=['NA'],encoding='utf8')
df3 = pd.read_csv(r"E:\MMA\MMA869\team project\dataset\TEAM Project\4910797b-ee55-40a7-8668-10efd5c1b960.csv",na_values=['NA'],encoding='utf8')


## join df1 and df3 it seems like df2 is testing value
df = pd.merge(df1,df3, how ='inner',on='id')
df.columns
df.info()
del df1,df3

# drop dupolicate records
df = df.drop_duplicates(subset=df.columns[1:41])


df.loc[:,df.isnull().any()].isnull().sum()

# general check
df['status_group'][0].strip()
df['status_group']= df['status_group'].str.strip()
df['status_group'].value_counts()
df['status_group'].value_counts(normalize=True)


# create a column to binary the amount of it
df['amount_tsh_ind'] = np.where(df['amount_tsh']> 0, 1, 0)
df2['amount_tsh_ind'] = np.where(df2['amount_tsh']> 0, 1, 0)
df['amount_tsh_ind'].value_counts()
df['amount_tsh'] = np.log(df['amount_tsh'].astype(float)+1)
df2['amount_tsh'] = np.log(df2['amount_tsh'].astype(float)+1)



# even keep top 100, remaining group still capture 25.5% of total count,and each funder min has 84 rows in
df['funder'] = df['funder'].str.lower()
df['funder'] = df['funder'].str.replace('  ',' ')
df2['funder'] = df2['funder'].str.lower()
df2['funder'] = df2['funder'].str.replace('  ',' ')
df['funder']= np.where(df['funder'].str.contains("gov",na=False,case=False),"government",df['funder'])
df['funder']= np.where(df['funder'].str.contains("cou",na=False,case=False),"council",df['funder'])
df['funder']= np.where(df['funder'].str.contains("comm",na=False,case=False),"community",df['funder'])
df['funder']= np.where(df['funder'].str.contains("vis",na=False,case=False),"world vision",df['funder'])
df['funder']= np.where(df['funder'].str.contains("chu|catholic",na=False,case=False),"church",df['funder'])
df['funder']= np.where(df['funder'].str.contains("dani|dann",na=False,case=False),"danida",df['funder'])
df['funder']= np.where(df['funder'].str.contains("dw",na=False,case=False),"dew",df['funder'])
df['funder']= np.where(df['funder'].str.contains("vil",na=False,case=False),"village",df['funder'])
df['funder']= np.where(df['funder'].str.contains("ltd|corp",na=False,case=False),"company",df['funder'])
df['funder']= np.where(df['funder'].str.contains("school",na=False,case=False),"school",df['funder'])
df['funder']= np.where(df['funder'].str.contains("0|na|-",na=False,case=False),"other",df['funder'])
df['funder']= np.where(df['funder'].str.contains("kk",na=False,case=False),"KKKT",df['funder'])
df['funder']= np.where(df['funder'].str.contains("germ",na=False,case=False),"germany",df['funder'])
df['funder']= np.where(df['funder'].str.contains("kili",na=False,case=False),"kili water",df['funder'])
df['funder']= np.where(df['funder'].str.contains("hes",na=False,case=False),"hesawa",df['funder'])
df['funder']= np.where(df['funder'].str.contains("fin",na=False,case=False),"fini water",df['funder'])
df['funder']= np.where(df['funder'].str.contains("unisef|unicef|unice",na=False,case=False),"unicef",df['funder'])
funder_less_other = df.groupby('funder')['funder'].count()[df.groupby('funder')['funder'].count()<100].index.tolist()
df['funder'].replace(to_replace=funder_less_other,value='other',inplace=True)
df['funder'].fillna('other', inplace=True)

df2['funder']= np.where(df2['funder'].str.contains("gov",na=False,case=False),"government",df2['funder'])
df2['funder']= np.where(df2['funder'].str.contains("cou",na=False,case=False),"council",df2['funder'])
df2['funder']= np.where(df2['funder'].str.contains("comm",na=False,case=False),"community",df2['funder'])
df2['funder']= np.where(df2['funder'].str.contains("vis",na=False,case=False),"world vision",df2['funder'])
df2['funder']= np.where(df2['funder'].str.contains("chu|catholic",na=False,case=False),"church",df2['funder'])
df2['funder']= np.where(df2['funder'].str.contains("dani|dann",na=False,case=False),"danida",df2['funder'])
df2['funder']= np.where(df2['funder'].str.contains("dw",na=False,case=False),"dew",df2['funder'])
df2['funder']= np.where(df2['funder'].str.contains("vil",na=False,case=False),"village",df2['funder'])
df2['funder']= np.where(df2['funder'].str.contains("ltd|corp",na=False,case=False),"company",df2['funder'])
df2['funder']= np.where(df2['funder'].str.contains("school",na=False,case=False),"school",df2['funder'])
df2['funder']= np.where(df2['funder'].str.contains("0|na|-",na=False,case=False),"other",df2['funder'])
df2['funder']= np.where(df2['funder'].str.contains("kk",na=False,case=False),"KKKT",df2['funder'])
df2['funder']= np.where(df2['funder'].str.contains("germ",na=False,case=False),"germany",df2['funder'])
df2['funder']= np.where(df2['funder'].str.contains("kili",na=False,case=False),"kili water",df2['funder'])
df2['funder']= np.where(df2['funder'].str.contains("hes",na=False,case=False),"hesawa",df2['funder'])
df2['funder']= np.where(df2['funder'].str.contains("fin",na=False,case=False),"fini water",df2['funder'])
df2['funder']= np.where(df2['funder'].str.contains("unisef|unicef|unice",na=False,case=False),"unicef",df2['funder'])
funder_less_other = df2.groupby('funder')['funder'].count()[df2.groupby('funder')['funder'].count()<100].index.tolist()
df2['funder'].replace(to_replace=funder_less_other,value='other',inplace=True)

df2['funder'].fillna('other', inplace=True)
df2['funder'].value_counts()



t1 = pd.crosstab(df.funder,df.status_group, normalize='index')

# index for function, using 0.7, base rate for cuntion is 54.3%
l1= t1[t1['functional'] > 0.7].index.tolist()
df['funder_function_ind'] = np.where(df['funder'].isin(l1), 1, 0)
df2['funder_function_ind'] = np.where(df2['funder'].isin(l1), 1, 0)
df['funder_function_ind'].value_counts()
# index for function, using 0.5, base rate for cuntion is 38.4%
l1= t1[t1['non functional'] > 0.5].index.tolist()
df['funder_nonfunction_ind']  = np.where(df['funder'].isin(l1), 1, 0)
df2['funder_nonfunction_ind'] = np.where(df2['funder'].isin(l1), 1, 0)
df['funder_nonfunction_ind'].value_counts()





##
# 'date_recorded'
df['date_recorded'].describe()


df['year'] = pd.DatetimeIndex(df['date_recorded']).year
df['month'] = pd.DatetimeIndex(df['date_recorded']).month
df['weekday'] = pd.to_datetime(df['date_recorded']).dt.dayofweek
df['weekday_ind'] = np.where(df['weekday'] <=5,1,0)

df2['year'] = pd.DatetimeIndex(df2['date_recorded']).year
df2['month'] = pd.DatetimeIndex(df2['date_recorded']).month
df2['weekday'] = pd.to_datetime(df2['date_recorded']).dt.dayofweek
df2['weekday_ind'] = np.where(df2['weekday'] <=5,1,0)


###############
# numeric columns impute


# df['gps_height'] = df['gps_height'].replace(0,np.NaN)
# df['longitude'] = df['longitude'].replace(0, np.NaN)
# df['latitude'] = df['latitude'].replace(0, np.NaN)
# df2['gps_height'] = df2['gps_height'].replace(0,np.NaN)
# df2['longitude'] = df2['longitude'].replace(0, np.NaN)
# df2['latitude'] = df2['latitude'].replace(0, np.NaN)


# fill na with mean 
# df["latitude"].fillna(df.groupby(['region', 'district_code'])["latitude"].transform("mean"), inplace=True)
# df["longitude"].fillna(df.groupby(['region', 'district_code'])["longitude"].transform("mean"), inplace=True)
# df["longitude"].fillna(df.groupby(['region'])["longitude"].transform("mean"), inplace=True)
# df["gps_height"].fillna(df["gps_height"].mean(), inplace=True)

# # fill na with mean 
# df2["latitude"].fillna(df2.groupby(['region', 'district_code'])["latitude"].transform("mean"), inplace=True)
# df2["longitude"].fillna(df2.groupby(['region', 'district_code'])["longitude"].transform("mean"), inplace=True)
# df2["longitude"].fillna(df2.groupby(['region'])["longitude"].transform("mean"), inplace=True)
# df2["gps_height"].fillna(df2["gps_height"].mean(), inplace=True)



# even keep top 100, remaining group still capture 25.5% of total count,and each funder min has 84 rows in
df['installer'] = df['installer'].str.upper()
df['installer'] = df['installer'].str.replace('  ',' ')
df2['installer'] = df2['installer'].str.upper()
df2['installer'] = df2['installer'].str.replace('  ',' ')
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
df["latitude"].fillna(df.groupby(['region', 'district_code'])["latitude"].transform("mean"), inplace=True)
df["longitude"].fillna(df.groupby(['region', 'district_code'])["longitude"].transform("mean"), inplace=True)
df["longitude"].fillna(df.groupby(['region'])["longitude"].transform("mean"), inplace=True)
df2["latitude"].fillna(df2.groupby(['region', 'district_code'])["latitude"].transform("mean"), inplace=True)
df2["longitude"].fillna(df2.groupby(['region', 'district_code'])["longitude"].transform("mean"), inplace=True)
df2["longitude"].fillna(df2.groupby(['region'])["longitude"].transform("mean"), inplace=True)

df['longitude_v2'] = pd.cut(df['longitude'],  bins=[-1,34,38,41],labels=["low", "medium", "high"])
df2['longitude_v2'] = pd.cut(df2['longitude'], bins=[-1,34,38,41],labels=["low", "medium", "high"])
df['longitude'] = df['longitude'].mask(df['longitude'] > -0.1, inplace=True)
df2['longitude'] = df2['longitude'].mask(df2['longitude'] > -0.1, inplace=True)






df['population_v2'] = pd.cut(df['population'], bins=[-1,2,160,30500],labels=["low", "medium", "high"])
df2['population_v2'] = pd.cut(df2['population'], bins=[-1,2,160,30500],labels=["low", "medium", "high"])
df['population'] = np.log(df['population']+1)
df2['population'] = np.log(df2['population']+1)

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


















# l = ['basin','funder','installer_v2','quantity','quality_group','lga_v2','waterpoint_type_v2','extraction_type_v2',\
#      'extraction_type_class','extraction_type_group','source_type','public_meeting','permit','longitude_v2','management',\
#        'scheme_management_v2', 'population_v2', 'payment','region']

###########################
# Creating dummy variables:
df = pd.get_dummies(df, columns=['basin'],dummy_na=True)
df = pd.get_dummies(df, columns=['funder'],dummy_na=True)
df = pd.get_dummies(df, columns=['installer_v2'],dummy_na=True)
df = pd.get_dummies(df, columns=['quantity'],dummy_na=True) #quantity is exactly same as quantity_group
df = pd.get_dummies(df, columns=['quality_group'],dummy_na=True) # add in this actually lower the prediction
df = pd.get_dummies(df, columns=['lga_v2'],dummy_na=True)
df = pd.get_dummies(df, columns=['ward_v2'],dummy_na=True)
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
df2 = pd.get_dummies(df2, columns=['funder'],dummy_na=True)
df2 = pd.get_dummies(df2, columns=['installer_v2'],dummy_na=True)
df2 = pd.get_dummies(df2, columns=['quantity'],dummy_na=True)
df2 = pd.get_dummies(df2, columns=['quality_group'],dummy_na=True)
df2 = pd.get_dummies(df2, columns=['lga_v2'],dummy_na=True)
df2 = pd.get_dummies(df2, columns=['ward_v2'],dummy_na=True)
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
df = df.drop(labels=['source_type_nan','region_code','funder_adra', 'funder_african', 'funder_bsf', 'funder_ces (gmbh)',
       'funder_ces(gmbh)', 'funder_co', 'funder_community', 'funder_concern',
       'funder_concern world wide', 'funder_ded', 'funder_dfid', 'funder_dh',
       'funder_dmdd', 'funder_fw', 'funder_go',
       'funder_halmashauri ya wilaya sikonge', 'funder_he', 'funder_hsw',
       'funder_ir', 'funder_is', 'funder_isf', 'funder_jaica', 'funder_jica',
       'funder_ki', 'funder_kili water', 'funder_lamp',
       'funder_lawatefuka water supply', 'funder_lvia', 'funder_mission',
       'funder_muwsa', 'funder_nethalan', 'funder_no', 'funder_oikos e.afrika',
       'funder_oxfam', 'funder_oxfarm', 'funder_plan int', 'funder_private',
       'funder_rc', 'funder_roman', 'funder_ru', 'funder_rudep',
       'funder_rural water supply and sanitat', 'funder_shipo', 'funder_snv',
       'funder_swedish', 'funder_tardo', 'funder_tassaf', 'funder_unhcr',
       'funder_village', 'funder_w.b', 'funder_wateraid', 'funder_wsdp',
       'funder_wua', 'funder_wvt'],axis =1)
df2 = df2.drop(labels=['region_code'],axis =1)


#######################################################
# https://www.analyticsvidhya.com/blog/2021/08/complete-guide-on-how-to-use-lightgbm-in-python/
# LGBM

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics


# To define the input and output feature
x = df.select_dtypes(include=np.number).fillna(0).drop(labels='id',axis=1)
y = df.status_group
x_predict = df2.select_dtypes(include=np.number).fillna(0).drop(labels='id',axis=1)
#x_predict = pd.concat([x_predict,df.loc[:,l]],axis=1)
# train and test split
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)


################
# hyperparameter tunning
from sklearn.model_selection import GridSearchCV


# {'learning_rate': 0.12, 'min_child_samples': 70, 'n_estimators': 600} 0.806111443538909
model = lgb.LGBMClassifier(objective='multiclass',zero_as_missing=True,random_state=42)
params = {
           'learning_rate':[0.1,0.08,0.06],
           'num_iterations': [800,1000],
           'num_leaves':[70,80,100],
           'min_data_in_leaf':[500,400],

    }

search = GridSearchCV(model,params,scoring='accuracy',cv=5,verbose=20)
search = search.fit(x,y)

print(search.best_params_, search.best_score_)


# plot feature importance
# feat_imp = pd.Series(search.feature_importances_, index=x.columns)
# feat_imp.nlargest(30).plot(kind='barh', figsize=(8,10))


##########################################
# from hyperparameter tunning
# choose the final one
lgbm_cv_y_pred = search.best_estimator_.predict(x_predict)

data_pred = pd.DataFrame()
data_pred['id'] = df2['id']
data_pred['status_group'] = lgbm_cv_y_pred



###########################################################
# choose 
# train and test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)


model = lgb.LGBMClassifier(objective='multiclass',learning_rate=0.06,num_iterations=800,random_state=42,num_leaves=100,
                           zero_as_missing=True,min_data_in_leaf=200)
model.fit(x_train,y_train,eval_set=[(x_test,y_test),(x_train,y_train)],
          verbose=30,eval_metric='logloss',early_stopping_rounds=20)


print('Training accuracy {:.4f}'.format(model.score(x_train,y_train)))
print('Testing accuracy {:.4f}'.format(model.score(x_test,y_test)))


y_pred = model.predict(x_predict)

data_pred = pd.DataFrame()
data_pred['id'] = df2['id']
data_pred['status_group'] = y_pred

# feature importance
feat_imp = pd.Series(model.feature_importances_, index=x.columns)
feat_imp.nlargest(30).plot(kind='barh', figsize=(8,10))
feat_imp.nlargest(30).sort_values(ascending=False).plot(kind='barh', figsize=(8,10))
plt.title("Feature Importances")




#confusion matrix
confusion_matrix(y_test, model.predict(x_test), labels=["functional", "non functional", "functional needs repair"])










############################
# cross-validation
import time
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 5
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
data_pred['id'] = df2['id']
data_pred['status_group'] = y_pred_v2






# XGboost part from here

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score

x = df.select_dtypes(include=np.number).fillna(0)
y = np.where(df.status_group == 'functional', 1,   #when... then
                 np.where(df.status_group == 'non functional', 0,  #when... then
                  2))


# data_dmatrix = xgb.DMatrix(data=x,label=y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# classifier not regressor!!
xg_reg = xgb.XGBClassifier(objective ='multi:softprob', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 7, c = 10, n_estimators = 10,num_class=3)
xg_reg.fit(X_train,y_train)

train_preds = xg_reg.predict(X_train)
#train_preds =pd.DataFrame(train_preds).idxmax(axis=1)
test_preds = xg_reg.predict(X_test)
#test_preds = pd.DataFrame(test_preds).idxmax(axis=1)

print('Training accuracy {:.4f}'.format(accuracy_score(y_train, train_preds)))
print('Testing accuracy {:.4f}'.format(accuracy_score(y_test, test_preds)))



preds = xg_reg.predict(x_predict)
# return the column name with max value
xgb_y =pd.DataFrame(preds).idxmax(axis=1)
xgb_y = np.where(xgb_y == 1, 'functional',   #when... then
                 np.where(xgb_y == 0, 'non functional',  #when... then
                  'functional needs repair'))








##################################################################
# Random Forest

from sklearn.ensemble import RandomForestClassifier

# To define the input and output feature
x = df.select_dtypes(include=np.number).fillna(0).drop(labels='id',axis=1)
y = df.status_group
x_predict = df2.select_dtypes(include=np.number).fillna(0).drop(labels='id',axis=1)


# train and test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)


# hyperparameter tunning
model = RandomForestClassifier(random_state=42)
params = {
   #       'class_weight':('balanced',None),
          'max_depth':[25,30],
          'min_samples_split':[20,30],
          'n_estimators': [1000,1500]
    }
 
search = GridSearchCV(model,params,scoring='accuracy',cv=4,verbose=1)
search = search.fit(x,y)
print(search.best_params_, search.best_score_)

rf_y = search.predict(x_predict)


# hyperparameter tunning
model = RandomForestClassifier(random_state=42,n_estimators=1500,max_depth=30,min_samples_split=50)
params = {
    #      'class_weight':('balanced',None),
   #       'max_depth':[25,30],
   #       'min_samples_split':[20,30],
  #        'n_estimators': [1000,1500]
    }
 
search = GridSearchCV(model,params,scoring='accuracy',cv=4,verbose=1)
search = search.fit(x,y)
print(search.best_params_, search.best_score_)

rf_y = search.predict(x_predict)


# min_samples_split: The minimum number of samples required to split an internal node:
    
    
clf=RandomForestClassifier(random_state=42,n_estimators=1500, max_depth= 30,min_samples_split=30)
#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(x_train,y_train)

train_preds = clf.predict(x_train)
test_preds  = clf.predict(x_test)


print('Training accuracy {:.4f}'.format(accuracy_score(y_train, train_preds)))
print('Testing accuracy {:.4f}'.format(accuracy_score(y_test, test_preds)))

rf_y = clf.predict(x_predict)

data_pred = pd.DataFrame()
data_pred['id'] = df2['id']
data_pred['status_group'] = rf_y






#########################
# compare all the results
result = pd.read_csv(r"E:\MMA\MMA869\team project\dataset\TEAM Project\result.csv",na_values=['NA'],encoding='utf8')
#y_pred_v2 = pd.read_csv(r"E:\MMA\MMA869\team project\dataset\TEAM Project\results\water_pump_prediction_08158.csv",na_values=['NA'],encoding='utf8')


result = pd.DataFrame()
result['id'] = df2['id']
result['lgbm'] = y_pred
result['xgboost'] = xgb_y
result['rf_y'] = rf_y


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

