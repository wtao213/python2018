# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 16:08:51 2021

@author: 012790
"""
import pandas as pd
import numpy as np
from datetime import datetime, date

df = pd.read_csv(r'C:\Users\012790\Desktop\python_practise\TTC_delay_data_full_Aug05.csv',sep=',', na_values=['','NULL'])  
weather = pd.read_csv(r'C:\Users\012790\Desktop\python_practise\weatherstats_toronto_normal_daily.csv',sep=',', na_values=['','NULL'])

# change name
df = df.rename(columns = {"year":"Year"})
df = df.rename(columns ={"year" : "Year",
                        "Date" : "date" })


#conver to string
df['Min.Delay'].astype(str)
df['Min.Delay'].apply(str)

# convert Series to float16 type
s = s.astype(np.float16)

# convert all DataFrame columns to the int64 dtype
df = df.astype(int)

##################################################################
# data clean up
# check data type
print (df.dtypes)


# remove duplicates
df.drop_duplicates()

# remove duplicates by columns and have max in others
df.drop_duplicates(subset='Station')

t = df.groupby('Station', group_keys=False).apply(lambda x: x.loc[x.year.idxmax()])

# top/tail 10
df.head(10)
df.tail(10)

## look at the first 10
df['Station'][0:10]


#########
# sampling
df.sample(frac = 0.1)
df.sample(n=10)


# select by position
df.iloc[10:20]    # select by row
df.iloc[:,10:20]   # select by column


# select by value
df.nlargest(10, 'Min.Delay')
df.nsmallest(10, 'Min.Delay')


# filter dataframe by column name or column index
df.loc[:,'x2':'x4']
df.iloc[:,[1,2,5]]




# freq table
t = df['Station'].value_counts()
t.head(10)

t = sorted(df['Station'].value_counts(),reverse=True)

pd.crosstab(df['year'], df['Day_name'], dropna=False)


df.sort_values('Min.Delay',ascending =True)
df.sort_values('Min.Delay',ascending =True)

df.dropna()
df.dropna(how='all')


df.describe()
df.dropna().describe()
df['Min.Delay'].describe()
df['Min.Delay'].quantile([0.05,0.5,0.95])

## create column based on if else
df['hasimage'] = np.where(df['photos']!= '[]', True, False)

df['Line'].value_counts()

df['Line2'] = np.where(df['Line'].apply(lambda x: x in ['YU','BD','SRT']), 1,0)
df['Line2'].value_counts()



# filtering and check the length
len(df.loc[(df['Min.Delay'] > 0) & (df['Year'] > 2018)])
len(df[(df['Min.Delay'] > 0) & (df['Year'] > 2018)])

# calculate date to now

#Get today's date
today = date.today()

datetime.strptime('13/02/1990', "%d/%m/%Y").date()

# This function converts given date to age
def age(born):
    born = datetime.strptime(born, "%d/%m/%Y").date()
    today = date.today()
    return today.year - born.year - ((today.month,  today.day) < (born.month, born.day))
  
df['Age'] = df['DOB'].apply(age)



# round() ,digits default =0 if what 2 digits round(x,2)
# round down using int()
def age(born):
    born  = datetime.striptime(born,"%d/%m/%Y").date()
    today = date.today()
    delta = today - born
    return int(delta.days/365.25,0)

int(3.7)
###########
# simple plotting
df['Min.Delay'].describe()
df.loc[df['Min.Delay'] < 50,'Min.Delay'].plot.hist()



##################################################################
# data merge
pd.merge(df,weather,how = 'left', left_on='Date',right_on='date')





##################################################################
# data aggregate
a = df.groupby(by='Station')






data.groupby( ['month', 'item']).agg(
    {
        # Find the min, max, and sum of the duration column
        'duration': [min, max, sum],
        # find the number of network type entries
        'network_type': "count",
        # minimum, first, and number of unique dates
        'date': [min, 'first', 'nunique']
    }
)
      
      
# method 1:      
station = df.groupby(['Station']).agg(
    {
     'Min.Delay':[min,max,sum],
     'column_label' : "count",
     'Line' :"first"
     }
    )

station.describe()


# method 2:   
print(df.dtypes)

station = df.groupby(['Station'],as_index=False).agg(
    N = ('Min.Delay', len),
    Min_Delay_max =('Min.Delay', max),
    Min_Delay_sum =('Min.Delay', sum),
    precipitation_sum = ('precipitation_ind',sum)
    )

station['avg'] = station['Min_Delay_sum'] / station['N']
station['precipition_avg'] = station['precipitation_sum'] / station['N']

station.drop_duplicates()
station.drop_duplicates(subset=['Min_Delay_sum'])
station.duplicated().sum()



station.duplicated().sum()

# check duplicates
station['dup_ind'] = np.where(station.duplicated(),1,0)
station['dup_ind'].value_counts()



# pratise
df.groupby(['Station','Date'],as_index = False).agg(
    N = ('Station',len),
    Precipitation = ('precipitation_ind',max),
    Min_delay = ('Min.Delay',sum)
    )





#######################################
c1 = df.groupby('Station')['Min.Delay'].describe()
















