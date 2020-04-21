# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:26:29 2020

transfer:

@author: 012790
"""

import pandas as pd
import numpy as np
import csv

import re

# from csv import reader

## 
df = pd.read_csv(r"C:\Users\012790\Desktop\Demo11.csv",encoding='windows-1252',header = None)


t1= pd.read_csv(r"C:\Users\012790\Desktop\Demo11.csv",encoding='windows-1252',header = None,sep="':|,")



## step 1: remove punctuation like '' or u'',"


## iloc[i] to specify i th row
s= df.iloc[0]

#tt= s.replace('\'|u\'', '')




## convert talbe info to string
s1 = str(s)


## replace re.sub('[a-z]*@', 'ABC@', s)
t2= re.sub('\'|u\'','',s)

## re.split(pattern, string, maxsplit=0, flags=0)


tt = re.split(':|,',t2)


del s1,t2


########################################################################
## part 1: try to exlucde certain special char like ', u', {}

def clean(x):
    return re.sub('\'|u\'|{|}','',x)


## clean the data frame
df['new'] = df.iloc[:,0].apply(clean)


def split(x):
    return re.split(',',x)


t= df['new'].apply(split)






#####################################################################
## part 2:
## try 1, if match a name ,return value






s = df['new'][1]

# re.search("className", s)
m = re.search('u_discovered_manufacturer', s)
m.span()

type(m.span())

## test whether is none
m is None

## if not in
m = re.search('asdfgh', s)




# find the last position
m.span()[1]

# s.find(',',58)
s.find(',',m.span()[1])

a = s[59:73]


## now try to use look
## list of column names
## s is the string
del a,end,m,n,s,s1,start,t,t1,value

l = ['className','u_discovered_manufacturer','u_device_os','x_sclo_scilogic_region']



for n in l: 
   test = re.search(n, s)
   if test is None:
       break
   start = test.span()[1]
   end   = s.find(',',start) 
   
   print(n, s[start+1:end])




## now, if I find the value , assign it to dataframe
## create a new dataframe df2 with our new info in df
header_list = ['new','className','u_discovered_manufacturer','u_device_os','x_sclo_scilogic_region']

df2 = df.reindex(columns = header_list)    
df2['new'] = df['new'] 
   
   
    
## now test with function
l = ['className','u_discovered_manufacturer','u_device_os','x_sclo_scilogic_region']


## in this round, since we are doing row by row, so if didn't find, will return missing

for n in l:
    globals()['store%s' % l.index(n)] =[]
    for row in df2['new']:
        test = re.search(n,row)       
        if test is None:
           globals()['store%s' % l.index(n)].append("")
        else:
            start = test.span()[1]
            end   = row.find(',',start) 
   
            value = row[start+1:end]
            globals()['store%s' % l.index(n)].append(value)
    df2[n] = globals()['store%s' % l.index(n)]

    
    
       
re.search('u_discovered_manufacturer',df2['new'][0])
    



