# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 20:19:13 2021

@author: wanti
"""

# https://www.datacamp.com/community/tutorials/xgboost-in-python
# https://github.com/r4msi/DrivenData-PumpItUp/blob/master/XGboost_Lumping.md

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

del df1,df3

df.loc[:,df.isnull().any()].isnull().sum()
df2.loc[:,df2.isnull().any()].isnull().sum()


##
# drop dupolicate records
df = df.drop_duplicates(subset=df.columns[1:41])



# check the numeric columns
df.select_dtypes(include=np.number).columns
df.select_dtypes(exclude=np.number).columns

cor= df.corr()
chi2_contingency(df['ward'],df['installer'])
chi2_table(df['ward'],df['installer'])



# general check
df['status_group'][0].strip()
df['status_group']= df['status_group'].str.strip()
df['status_group'].value_counts()

# plot
df['status_group'].value_counts().plot(kind='bar', stacked=True)
plt.title("Status_group Distribution")
plt.xlabel("Status_group")
plt.ylabel("Count")


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


#################
#
df['population'].describe()


plt.hist(df.loc[(df.status_group == 'functional'),['population']], bins=50,color = "skyblue",ec='blue',range=(0,12),alpha =0.5)
plt.hist(df.loc[(df.status_group == 'non functional'),['population']], bins=50,color = "red",ec='red',range=(0,12),alpha =0.5)
plt.hist(df.loc[(df.status_group == 'functional needs repair'),['population']], bins=50,color = "green",ec='red',range=(0,12),alpha =0.5)
plt.xlim(0,12)
plt.title("population by status_group Distribution")
plt.xlabel("population")
plt.ylabel("Count")
# add in vertial line
#plt.vlines(x=317,  colors='red', ls=':',ymin=0, ymax=25,lw=2, label=' Average = 317')
plt.legend( loc='upper right')
plt.show()


####################################
# construction year
df.columns

df['construction_year'] = df['construction_year'].replace(0, np.NaN)
agg_tips = df.groupby(['construction_year','status_group'])['id'].count().unstack().fillna(0)
agg_tips.plot(kind='bar', stacked=True)
plt.title("construction_year by status_group Distribution")
plt.xlabel("construction_year"，rotation=45)
plt.ylabel("Count"）







#plot version 2
sb.catplot(x="status_group", y="amount_tsh", data=df)

# create a column to binary the amount of it
df['amount_tsh_ind'] = np.where(df['amount_tsh']> 0, 1, 0)
df['amount_tsh_ind'].value_counts()
df['amount_tsh'] = np.log(df['amount_tsh'].astype(float)+1)


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


funder_government = df[df['funder'].str.contains("gov",na=False,case=False)]['funder'].unique().tolist()
df['funder'].replace(to_replace=funder_government,value='government',inplace=True)

# df['funder']=[re.sub(r"GOV","GOVERNMENT", str(x)) for x in df['funder']] # this only change match parten, not whole string

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
df['funder'].value_counts()












##
# 'date_recorded'
df['date_recorded'].describe()

df['date_recorded'].value_counts().head(50).plot(kind='bar')
df['date_recorded'].value_counts()


df['year'] = pd.DatetimeIndex(df['date_recorded']).year
df['month'] = pd.DatetimeIndex(df['date_recorded']).month
df['weekday'] = pd.to_datetime(df['date_recorded']).dt.dayofweek



##
# 'gps_height'
df.columns

df['gps_height'].describe()
df['gps_height'].quantile(np.arange(0.9, 1.0, 0.02))


plt.hist(df.loc[(df.status_group == 'functional'),['gps_height']], bins=50,color = "skyblue",ec='blue',range=(-100,2000),alpha =0.5)
plt.hist(df.loc[(df.status_group == 'non functional'),['gps_height']], bins=50,color = "coral",ec='red',range=(-100,2000),alpha =0.5)
plt.hist(df.loc[(df.status_group == 'functional needs repair'),['gps_height']], bins=50,color = "lightgreen",ec='green',range=(-100,2000),alpha =0.5)
plt.xlim((-100,2000))
plt.title("gps_height by outcome Distribution")
plt.xlabel("gps_height")
plt.ylabel("Count")
# add in vertial line
#plt.vlines(x=317,  colors='red', ls=':',ymin=0, ymax=25,lw=2, label=' Average = 317')
plt.legend( loc='upper right')
plt.show()


# why the 0 bin so big? does 0 mean missing?

df['gps_height'].hist(by=df.status_group,bins=50,color = "skyblue",ec='blue',range=(-100,2000))
sb.catplot(x="status_group", y='gps_height', data=df[df['gps_height'] !=0])

# 20438 is 0, which is 34.4%
df[df['gps_height']==0]

df.loc[df['gps_height']==0,'status_group'].value_counts(normalize=True)
df.loc[df['gps_height']>1600,'status_group'].value_counts(normalize=True)
df.loc[df['gps_height']>1600,'status_group'].value_counts()
df['status_group'].value_counts(normalize=True)


df['gps_height_0_ind'] = np.where(df['gps_height']==0,1,0)
df['gps_height_high_ind'] =np.where(df['gps_height']>1600,1,0)


######################
# lets focus on category first
df.columns

df['installer'].describe()
df['installer'].value_counts()
df['installer'].value_counts().head(100).cumsum()
df['installer'].value_counts(normalize=True).head(100).cumsum()


df['installer'] = df['installer'].str.upper()
df['installer'] = df['installer'].str.replace('  ',' ')
# choose top 100 which capture 0.78
df.groupby(["status_group", "installer"]).size().reset_index(name="Time")

t1 = pd.crosstab(df.installer,df.status_group, normalize='columns')

# even keep top 100, remaining group still capture 18.5% of total count,and each funder min has 84 rows in
l1 = df['installer'].value_counts().head(100).index.tolist()
df['installer_v2'] = np.where(df['installer'].isin(l1), df['installer'], 'other')

# then look at the result by installer
t1 = pd.crosstab(df.installer_v2,df.status_group, normalize='index')

# index for function, using 0.7, base rate for cuntion is 54.3%
l1= t1[t1['functional'] > 0.7].index.tolist()
df['installer_function_ind'] = np.where(df['installer_v2'].isin(l1), 1, 0)
df['installer_function_ind'].value_counts()
# index for function, using 0.5, base rate for cuntion is 38.4%
l1= t1[t1['non functional'] > 0.5].index.tolist()
df['installer_nonfunction_ind'] = np.where(df['installer_v2'].isin(l1), 1, 0)
df['installer_nonfunction_ind'].value_counts()




######################
# 'quantity'
df.columns

df['quantity'].describe()
df['quantity'].value_counts()


######################
# 'quantity_group'
df.columns

df['quantity_group'].describe()
df['quantity_group'].value_counts()


######################
# 'lga'
df['lga'].describe()
df['lga'].value_counts()
df['lga'].value_counts().head(100).cumsum()
df['lga'].value_counts(normalize=True).head(100).cumsum()


l1 = df['lga'].value_counts().head(100).index.tolist()
df['lga_v2'] = np.where(df['lga'].isin(l1), df['lga'], 'other')



############
#
df['management'].describe()
df['management'].isnull().sum()


df['management_group'].describe()
df['management_group'].value_counts()
df['management_group'].isnull().sum()


##
# latitude longitude


df['longitude'].describe()
df['longitude'].isnull().sum()

df['longitude_v2'] = pd.qcut(df['longitude'], q=3)
pd.qcut(df['longitude'], q=3).value_counts(dropna=False)
df2['longitude_v2'] = pd.cut(df2['longitude'], bins=[-0.001,33.761,36.67,40.345])
pd.cut(df2['longitude'], bins=[-0.001,33.761,36.67,40.345]).value_counts(dropna=False)


# population
df['population_v2'] = pd.cut(df['population'], bins=[-1,2,160,30500],labels=["low", "medium", "high"])
df2['population_v2'] = pd.cut(df2['population'], bins=[-1,2,160,30500],labels=["low", "medium", "high"])




#  [scheme_management]
df['scheme_management'].describe()
df['scheme_management'].value_counts(dropna=False)

df['scheme_management_v2'] = np.where(df['scheme_management'].isnull(),'Other',
         np.where(df['scheme_management']=='None','Other',df['scheme_management']))

df['scheme_management_v2'].value_counts()





# extraction_type
df['extraction_type'].describe()
df['extraction_type'].value_counts(dropna=False)

df['extraction_type_v2'] = np.where(df['extraction_type'].isin(['other - mkulima/shinyanga','climax','walimi']),'Other',df['extraction_type'])

df['extraction_type_v2'].value_counts(dropna=False)





##
df['waterpoint_type_v2']  = np.where(df['waterpoint_type'] =='dam','cattle trough',df['waterpoint_type'])
df2['waterpoint_type_v2'] = np.where(df2['waterpoint_type'] =='dam','cattle trough',df2['waterpoint_type'])


df['waterpoint_type_v2'].value_counts(dropna=False)

#
df['district_code'] = df['district_code'].astype(str)
df['district_code'].describe()
df['district_code'].value_counts()
df['district_code_v2'] = np.where(df['district_code'].isin(['60','0','80','67']),'Other',df['district_code'])
df['district_code_v2'].value_counts()





###############################
# ward scheme_name

# display 
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

df['ward'].describe()
df['ward'].value_counts(dropna =False)

df['ward'] = df['ward'].str.upper()
df['ward'] = df['ward'].str.replace('  ',' ')




############################################
# numeric impute
df.columns
# for gps_height longitude , using median in the region code to replace the 0
df['gps_height'] = df['gps_height'].replace(0,np.NaN)
df['longitude'] = df['longitude'].replace(0, np.NaN)
df['latitude'] = df['latitude'].replace(0, np.NaN)

t1 = df.groupby(['region'],as_index = False).agg(
    gps_height_med = pd.NamedAgg(column='gps_height', aggfunc= 'median'),
    longitude_med  = pd.NamedAgg(column='longitude', aggfunc= 'median'),
    latitude_med   = pd.NamedAgg(column='latitude', aggfunc= 'median')    
    )

df = pd.merge(df,t1,how='left',on='region')

df['gps_height'] = np.where(df['gps_height'].isnull,df['gps_height_med'],df['gps_height'])
df['longitude'] = np.where(df['longitude'].isnull,df['longitude_med'],df['longitude'])
df['latitude'] = np.where(df['latitude'].isnull,df['latitude_med'],df['latitude'])
df['gps_height'] = df['gps_height'].replace(np.NaN,0)



#######################
# export to csv
df.to_csv(r"C:\Users\012790\Desktop\water_pump\water_pump.csv", index=False)









































## get numeric columns

df.select_dtypes(include=np.number)

df.columns
df.info()

# Creating dummy variables:
df = pd.get_dummies(df, columns=['quantity'])
df = pd.get_dummies(df, columns=['quantity_group'])
df = pd.get_dummies(df, columns=['lga'])
df = pd.get_dummies(df, columns=['waterpoint_type'])
df = pd.get_dummies(df, columns=['extraction_type'])
df = pd.get_dummies(df, columns=['extraction_type_class'])
df = pd.get_dummies(df, columns=['source_type'])
df = pd.get_dummies(df, columns=['public_meeting'])


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

x = df.select_dtypes(include=np.number)
y = np.where(df.status_group == 'functional', 1,   #when... then
                 np.where(df.status_group == 'non functional', 0,  #when... then
                  2))
# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

data_dmatrix = xgb.DMatrix(data=x,label=y)


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
print('Testing accuracy {:.4f}'.format(model.score(x_test,y_test)))# import data
