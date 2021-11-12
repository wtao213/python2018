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
        'Sum of Promise Amount':['mean'],
        'Broken': [sum],
        'Closed': [sum],
        'Open': [sum],
        'Kept':  [sum]
    }
)

df = df.droplevel(1, axis=1)  
df= df.set_axis(['Dialer Id', 'N', 'Number of RPC','Number of PTP','Average Payment Amount','Total Payment Amount','Average Promise Amount','Broken','Closed','Open','Kept'], axis=1)


df['PTP/RPC%'] =  df['Number of PTP']*100 / df['Number of RPC']
df['%of promises Kept'] = df['Kept'] *100 / (df['Kept'] + df['Broken'] )

#
df2= df.loc[:,['Dialer Id','Number of RPC','Number of PTP','PTP/RPC%','Average Promise Amount','%of promises Kept','Total Payment Amount']]
df2= df2.sort_values(by=['Total Payment Amount'],ascending=False)
df2.to_csv(r"E:\MMA\MMA841 Ops\A2\Q1_2.csv",index=False,sep=',')

del df2


############################
# 3) Mike expected that RPC (right party connect) rates would be fairly consistent across reps if they were all following the company script
time.columns
time['RPC_rate'] = time['Right Party Connect Count'] / time['Call Count']


time['RPC_rate'].describe()

time['RPC_rate'].mean()
time['RPC_rate'].std()

##
UCL = time['RPC_rate'].mean() + 3*time['RPC_rate'].std()
LCL = time['RPC_rate'].mean() - 3*time['RPC_rate'].std()


UCL
LCL

# 7 reps out of the boundry, which is 4.69%
time.loc[(time['RPC_rate'] > UCL) | (time['RPC_rate'] < LCL)]



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


#Call Count
df2['Call Count'].describe()


plt.hist(df2['Call Count'], bins=50,color = "skyblue",ec='blue')
plt.title("Dialer's Call Count Distribution")
plt.xlabel("Call Count")
plt.ylabel("Dialer count")
# add in vertial line
plt.vlines(x=2346,  colors='red', ls=':',ymin=0, ymax=15,lw=2, label='Call Count Average = 2346')
plt.legend( loc='upper right')
plt.show()






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
plt.plot(df2['Total Payment Amount'], df2['Call Count'], 'o', color='lightblue')
plt.title("Payment vs Call Count")
plt.xlabel("Total Payment Amount $")
plt.ylabel("Call Count")



# Number of RPC vs Call Count
plt.plot(df2['Number of RPC'], df2['Call Count'], 'o', color='lightblue')
plt.title("Number of RPC vs Call Count")
plt.xlabel("Number of RPC")
plt.ylabel("Call Count")


# Payment vs Number of RPC
plt.plot(df2['Total Payment Amount'], df2['Number of RPC'], 'o', color='lightblue')
plt.title("Payment vs Number of RPC")
plt.xlabel("Total Payment Amount $")
plt.ylabel("Number of RPC")




# 2d heat map, no need for this, scartter plot works
tb = pd.pivot_table(df2.loc[:,['Sum of Payment Amount','Call Count']])
tb = pd.pivot_table(df2,values='N' ,index=['Sum of Payment Amount'],columns =['Call Count'],aggfunc= 'count')

sb.heatmap(tb)






















