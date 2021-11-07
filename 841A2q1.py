# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 20:34:59 2021

@author: wanti
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb



rpc = pd.read_excel(r"E:\MMA\MMA841 Ops\A2\om-1220x.xlsx", sheet_name="RPCs",na_values=['NA',''],skiprows = 5,nrows=25075, usecols = 'A:N')

time = pd.read_excel(r"E:\MMA\MMA841 Ops\A2\om-1220x.xlsx", sheet_name="Time",na_values=['NA',''],skiprows = 5,nrows=149,usecols = 'A:E')



#######################################
# 1. 2) aggregate the data


df = rpc.groupby(['Dialer Id'],as_index=False).agg(
    {
        # Find the min, max, and sum of the duration column
        'Dial Date': "count",
        'RPC': [sum],
        # find the number of network type entries
        
        'PTP': [sum],
        'Sum of Payment Amount': ['mean',sum],
        'Broken': [sum],
        'Closed': [sum],
        'Open': [sum],
        'Kept':  [sum]
    }
)

df = df.droplevel(1, axis=1)  
df= df.set_axis(['Dialer Id', 'N', 'Number of RPC','Number of PTP','Average Payment Amount','Sum of Payment Amount','Broken','Closed','Open','Kept'], axis=1)


df['PTP/RPC%'] =  df['Number of PTP']*100 / df['Number of RPC']
df['%of promises Kept'] = df['Kept'] *100 / (df['Kept'] + df['Broken'] )



############################
# 3) Mike expected that RPC (right party connect) rates would be fairly consistent across reps if they were all following the company script
time.columns
time['RPC_rate'] = time['Right Party Connect Count'] / time['Call Count']


time['RPC_rate'].describe()

# can be use either in 3) or 4)
plt.hist(time['RPC_rate'], bins=40, range = (0, 0.2),color = "skyblue",ec='blue')
plt.title("Dialer's RPC Rate Distribution")
plt.xlabel("RPC rate")
plt.ylabel("Dialer count")
plt.vlines(x=0.0748,  colors='red', ls=':',ymin=0, ymax=30,lw=2, label='RPC Rate Average = 7.48%')
plt.legend( loc='upper right')
plt.show()





############################
# 4) anything you can tell

# get the full data frame

df2 = pd.merge(df,time,left_on='Dialer Id',right_on='Dialer ID', how='right')

df2.columns
# promise rate for dialer
df['PTP/RPC%'].describe()


plt.hist(df['PTP/RPC%'], bins=25,color = "skyblue",ec='blue')
plt.title("Dialer's PTP/RPC% Distribution")
plt.xlabel("PTP/RPC%")
plt.ylabel("Dialer count")
# add in vertial line
plt.vlines(x=21.88,  colors='red', ls=':',ymin=0, ymax=25,lw=2, label='PTP/RPC Average = 21.88%')
plt.legend( loc='upper right')
plt.show()



######################
#


# Payment vs Call Count
plt.plot(df2['Sum of Payment Amount'], df2['Call Count'], 'o', color='lightblue')
plt.title("Payment vs Call Count")
plt.xlabel("Sum of Payment Amount $")
plt.ylabel("Call Count")



# Number of RPC vs Call Count
plt.plot(df2['Number of RPC'], df2['Call Count'], 'o', color='lightblue')
plt.title("Number of RPC vs Call Count")
plt.xlabel("Number of RPC")
plt.ylabel("Call Count")


# Payment vs Number of RPC
plt.plot(df2['Sum of Payment Amount'], df2['Number of RPC'], 'o', color='lightblue')
plt.title("Payment vs Number of RPC")
plt.xlabel("Sum of Payment Amount $")
plt.ylabel("Number of RPC")




# 2d heat map, no need for this, scartter plot works
tb = pd.pivot_table(df2.loc[:,['Sum of Payment Amount','Call Count']])
tb = pd.pivot_table(df2,values='N' ,index=['Sum of Payment Amount'],columns =['Call Count'],aggfunc= 'count')

sb.heatmap(tb)






















