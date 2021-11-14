# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 09:01:30 2021

@author: wanti
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime as DT
from datetime import timedelta


########
df = pd.read_excel(r"E:\MMA\MMA841 Ops\topic5\05 EXCEL model and data -- Wells Fargo.xlsx", sheet_name="Demand and Solar Output",na_values=['NA',''])

df.columns
#
df.dtypes()
df['Date/Time'].dtypes()



df['Dates'] = pd.to_datetime(df['Date/Time']).dt.date
df['Time'] = pd.to_datetime(df['Date/Time']).dt.time




###
#   aggregate

date = df.groupby(['Dates'],as_index=False).agg(
    {
        # Find the min, max, and sum of the duration column
        'Date/Time': "count",
        'Electricity Demand for the Branch (kW)': [sum],     
        'Solar System Output (kW)': [sum]

    }
)

date = date.droplevel(1, axis=1) 

date['Dates'][5].strftime('%A')


# can be use either in 3) or 4)

plt.plot(date['Dates'], date['Electricity Demand for the Branch (kW)'],label='Electricity Demand for the Branch (kW)')
plt.plot(date['Dates'], date['Solar System Output (kW)'],label='Solar System Output (kW)')
#plt.legend( loc='upper right')
plt.legend(loc=(1.04,0))
plt.title("Demand vs Output by Date (kW)")
plt.xlabel("Date")
plt.ylabel("kW")
plt.show()


###############################################
#   aggregate

time = df.groupby(['Time'],as_index=False).agg(
    {
        # Find the min, max, and sum of the duration column
        'Date/Time': "count",
        'Electricity Demand for the Branch (kW)': [sum],     
        'Solar System Output (kW)': [sum]

    }
)

time = time.droplevel(1, axis=1) 
xtickes = np.arange(0, len(time['Time'])+1, 8)



# can be use either in 3) or 4)
time['Time']=time['Time'].astype(str)

plt.plot(time['Time'], time['Electricity Demand for the Branch (kW)'],label='Electricity Demand for the Branch (kW)')
plt.plot(time['Time'], time['Solar System Output (kW)'],label='Solar System Output (kW)')
#plt.legend( loc='upper right')
plt.legend(loc=(1.04,0))
plt.xticks(time['Time'][xtickes[0:11]],rotation=45)
plt.title("Demand vs Output by Time (kW)")
plt.xlabel("Time")
plt.ylabel("kW")
plt.show()
