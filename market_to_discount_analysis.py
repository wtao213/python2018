# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:53:48 2018

@author: wtao
"""

import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd

# import data set
data = pd.read_csv(r"C:\Users\wtao\Desktop\market to discount\Output_Cross_Shop_2016.txt",sep="|")

data2017 = pd.read_csv(r"C:\Users\wtao\Desktop\market to discount\Output_Cross_Shop_2017.txt",sep="|")
# replace missing value with zeros
df = data.fillna(0)
df17 = data2017.fillna(0)

# delete your record to save memory
del data
del data2017

# add colomn for discount, market and total for 2016
df['total_sales'] = df['SS_Sales'] + df['NF_Sales'] + df.FR_Sales + df.LBL_Sales + df.VM_Sales + df.YIG_Sales + df.Z_Sales 

df['total_Txn'] = df['SS_Txn'] + df['NF_Txn'] + df.FR_Txn + df.LBL_Txn + df.VM_Txn + df.YIG_Txn + df.Z_Txn 

# don't include fortino FR in the market sales

df['Market_sales'] =  df.LBL_Sales + df.VM_Sales + df.YIG_Sales + df.Z_Sales

df['Discount_sales'] = df['SS_Sales'] + df['NF_Sales']

df['Market_Txn'] = df.LBL_Txn + df.VM_Txn + df.YIG_Txn + df.Z_Txn 

df['Discount_Txn'] = df['SS_Txn'] + df['NF_Txn']

df['Market_pen'] = df.Market_sales / df.total_sales



# add colomn for discount, market and total for 2017
df17['total_sales'] = df17['SS_Sales'] + df17['NF_Sales'] + df17.FR_Sales + df17.LBL_Sales + df17.VM_Sales + df17.YIG_Sales + df17.Z_Sales 

df17['total_Txn'] = df17['SS_Txn'] + df17['NF_Txn'] + df17.FR_Txn + df17.LBL_Txn + df17.VM_Txn + df17.YIG_Txn + df17.Z_Txn 

# don't include fortino FR in the market sales

df17['Market_sales'] =  df17.LBL_Sales + df17.VM_Sales + df17.YIG_Sales + df17.Z_Sales

df17['Discount_sales'] = df17['SS_Sales'] + df17['NF_Sales']

df17['Market_Txn'] = df17.LBL_Txn + df17.VM_Txn + df17.YIG_Txn + df17.Z_Txn 

df17['Discount_Txn'] = df17['SS_Txn'] + df17['NF_Txn']

df17['Market_pen'] = df17.Market_sales / df17.total_sales


# delete columns in data frame
df.drop(['SS_Sales', 'NF_Sales', 'SS_Txn','NF_Txn','LBL_Sales','VM_Sales','YIG_Sales','Z_Sales',
                'LBL_Txn', 'VM_Txn','YIG_Txn','Z_Txn'], axis=1, inplace=True)

df17.drop(['SS_Sales', 'NF_Sales', 'SS_Txn','NF_Txn','LBL_Sales','VM_Sales','YIG_Sales','Z_Sales',
                'LBL_Txn', 'VM_Txn','YIG_Txn','Z_Txn'], axis=1, inplace=True)

#merge the data set by customer id
 result = pd.merge(df, df17, how='inner', on=['MBRSHIP_ID'])


del df
del df17
# x is year2016, y is year 2017
customer = result[(result.total_sales_x > 500)&(result.total_sales_y> 500)]

#market change
customer['market_change'] = customer.Market_pen_x - customer.Market_pen_y

customer = customer[(customer.market_change > 0.1)&(customer.total_sales_x<20000)&(customer.total_sales_y<20000)]


#output dataset
customer.to_csv(r"C:\Users\wtao\Desktop\market to discount\customer.txt",sep="|")
