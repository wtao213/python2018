# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:19:01 2021

@author: 012790
"""
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

df1 = pd.read_csv(r"C:\Users\012790\Desktop\water_pump\0bf8bc6e-30d0-4c50-956a-603fc693d966.csv",na_values=['NA',''],encoding='utf8')
df2 = pd.read_csv(r"C:\Users\012790\Desktop\water_pump\702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv",na_values=['NA',''],encoding='utf8')
df3 = pd.read_csv(r"C:\Users\012790\Desktop\water_pump\4910797b-ee55-40a7-8668-10efd5c1b960.csv",na_values=['NA',''],encoding='utf8')



## join df1 and df3 it seems like df2 is testing value
df = pd.merge(df1,df3, how ='inner',on='id')
df.columns
df.info()
del df1,df3

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

df['date_recorded'].value_counts().head(50).plot(kind='bar')
df['date_recorded'].value_counts()


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























# Creating dummy variables:
df = pd.get_dummies(df, columns=['quantity'])
df = pd.get_dummies(df, columns=['quantity_group'])
df = pd.get_dummies(df, columns=['lga'])
df = pd.get_dummies(df, columns=['waterpoint_type'])
df = pd.get_dummies(df, columns=['extraction_type'])
df = pd.get_dummies(df, columns=['extraction_type_class'])
df = pd.get_dummies(df, columns=['source_type'])
df = pd.get_dummies(df, columns=['public_meeting'])





df2 = pd.get_dummies(df2, columns=['quantity'])
df2 = pd.get_dummies(df2, columns=['quantity_group'])
df2 = pd.get_dummies(df2, columns=['lga'])
df2 = pd.get_dummies(df2, columns=['waterpoint_type'])
df2 = pd.get_dummies(df2, columns=['extraction_type'])
df2 = pd.get_dummies(df2, columns=['extraction_type_class'])
df2 = pd.get_dummies(df2, columns=['source_type'])
df2 = pd.get_dummies(df2, columns=['public_meeting'])

#######################################################
# https://www.analyticsvidhya.com/blog/2021/08/complete-guide-on-how-to-use-lightgbm-in-python/
# LGBM

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics


# To define the input and output feature
x = df.select_dtypes(include=np.number)
y = df.status_group
x_test = df.select_dtypes(include=np.number)

# train and test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)


model = lgb.LGBMClassifier(learning_rate=0.09,max_depth=-7,random_state=42,min_child_samples=50)
model.fit(x_train,y_train,eval_set=[(x_test,y_test),(x_train,y_train)],
          verbose=20,eval_metric='logloss')


print('Training accuracy {:.4f}'.format(model.score(x_train,y_train)))
print('Testing accuracy {:.4f}'.format(model.score(x_test,y_test)))


y_pred = model.predict(x_test)
y_pred = pd.Series(y_pred)

















