# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:34:11 2020

@author: 012790
"""

################################################################
## using sentimental analyasis to check the real time feedback
## vanderSentiment is used
## example linked below:
## https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f



## import the packages
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()



## funtion to test
## compound rate is lexicon ratings which between -1(extreme negative) and +1 (extreme positive)
## rate of neg, neu, pos sum up to 1.

## overall  positive sentiment: compound score >= 0.05
##          neutral sentiment: -0.05 < compound score < 0.05
##          negative sentiment: compound score <= -0.05
## https://github.com/cjhutto/vaderSentiment#about-the-scoring detail explaination

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))
    
    
sentiment_analyzer_scores("The phone is super cool.")

sentiment_analyzer_scores("Steve is one of the very best in the Senate. He is competing against a failed Democrat Presidential candidate who never got close to 1%, and was sent packing. Had to be talked into this run by Schumer. Bad for 2nd A! I strongly Endorse")




#########################################################
## import data
## try to testing on the data we arleady have.
df = pd.read_csv(r"C:\Users\012790\Desktop\survey\cus_full_v2.csv",encoding='windows-1252')


## look at the text countent, and get an idea of it's sentimental
## things want to know: 1.how many comment today?
## 2.how many of them overall pos? neg? neural?

a = df['improve_area']

## count how many of them are missing, 2231 out of 3859 is null
a.isnull().sum()

## need to remove missing value first and then run the function,keep id so could merge back
###### need to find easier way to do so 
del df2
df2 = df[['PrimaryClientID','improve_area']]
df2 = df2.dropna()

## apply a function to each row of the columns(series)
##df2['score']= sentiment_analyzer_scores(df2['improve_area'])


score = analyser.polarity_scores(df2['improve_area'].iloc[0])


## show the n th row of the dict list(my_dict.keys())[0]
## but we want to get the value of a certain key
score['compound']

score_compound = analyser.polarity_scores(df2['improve_area'].iloc[0])['compound']


###############
## finally work!!! this one is the original version with dictionary return
def get_sentiment(row):
 sentiment_score = analyser.polarity_scores(row)
 positive = sentiment_score['pos'] 
 negative = sentiment_score['neg'] 
 compound  = sentiment_score['compound']
 return positive,negative,compound


######################
 ## update version to specif directly, instead of a dictionary return

def get_sentiment(row,**kwargs):
 sentiment_score = analyser.polarity_scores(row)
 positive = sentiment_score['pos'] 
 negative = sentiment_score['neg'] 
 neural = sentiment_score['neu'] 
 compound = sentiment_score['compound']
 if kwargs['k'] == 'pos':
      return positive
 elif kwargs['k'] == 'neg':
      return negative
 elif kwargs['k'] == 'neu':
      return neural
 else:
      return compound 



df2['pos_score'] = df2.improve_area.apply(get_sentiment,k='pos')
df2['neg_score'] = df2.improve_area.apply(get_sentiment, k='neg')
df2['neu_score'] = df2.improve_area.apply(get_sentiment,k='neu')
df2['compound'] = df2.improve_area.apply(get_sentiment, k='compound')




## merge back to original dataframe
full = pd.merge(df,df2[['PrimaryClientID','pos_score','neg_score','neu_score','compound']],on='PrimaryClientID',how='left')

## export to excel
full.to_csv(r"C:\Users\012790\Desktop\survey\cus_full_v3.csv",sep=",")











