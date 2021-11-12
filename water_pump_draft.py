# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 20:19:13 2021

@author: wanti
"""

# https://www.datacamp.com/community/tutorials/xgboost-in-python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# import data

df1 = pd.read_csv(r"E:\MMA\MMA869\team project\dataset\TEAM Project\0bf8bc6e-30d0-4c50-956a-603fc693d966.csv",na_values=['NA',''],encoding='utf8')
df2 = pd.read_csv(r"E:\MMA\MMA869\team project\dataset\TEAM Project\702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv",na_values=['NA',''],encoding='utf8')
df3 = pd.read_csv(r"E:\MMA\MMA869\team project\dataset\TEAM Project\4910797b-ee55-40a7-8668-10efd5c1b960.csv",na_values=['NA',''],encoding='utf8')


## join df1 and df3 it seems like df2 is testing value

df = pd.merge(df1,df3, how ='inner',on='id')
df.columns

# general check
df['status_group'].value_counts()

df['funder'].value_counts().head(15)

df['gps_height'].value_counts(bins=10,ascending=True,dropna=False)

df['installer'].value_counts().head(15)


df.loc[:,['status_group','funder']]

pd.crosstab(df['funder'],df['status_group'])