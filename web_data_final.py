# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 20:40:19 2021

@author: wanti
"""
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import re
import numpy as np


################################################
#
driver = webdriver.Chrome(executable_path="C:\\Users\\wanti\\AppData\\Local\\chromedriver.exe")

url = 'https://www.wunderground.com/history/monthly/us/il/chicago/KMDW/date/2015-1'

# get web
driver.get(url)

ct = driver.find_elements_by_xpath('//*[@id="inner-content"]/div[2]/div[1]/div[5]/div[1]/div/lib-city-history-observation/div/div[2]/table/tbody/tr')

## check your output
print(ct[0].text) 

# save ouput to string
a = ct[0].text

## testing how to split
# split the string to a list of string
a = re.split('\n', a)


final_list = [None] *32
 
for i in range(7):
    globals()["l" + str(i)] = a[i*32:(i+1)*32]
    globals()["l" + str(i)] = [re.split('\s',x) for x in globals()["l" + str(i)]]
    
    final_list = np.column_stack([final_list,globals()["l" + str(i)]])
    del globals()["l" + str(i)]
 
    
 # export to csv
df = pd.DataFrame(data = final_list[1:,:], columns = final_list[0,:])
 


df.to_csv("C:\\Users\\wanti\\Desktop\\web_output.csv")  