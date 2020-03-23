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
import GetOldTweets3 as got
import csv

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






################################################
##      get twittwe from the web
##  https://pypi.org/project/GetOldTweets3/
##  

import GetOldTweets3 as got


## search by user
tweetCriteria = got.manager.TweetCriteria().setUsername("barackobama whitehouse")\
                                           .setMaxTweets(2)\
                                           .setLang('en')
                                        
                                           
tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]
print(tweet.text)

## search by query
## using " xxx OR xxx" to apply multiple key words
## during 2020-03-01 to 2020-03-07 there are90 tweets about questrade
## during 2020-03-16 to 2020-03-22 there are 980 tweets about questrade..
## keep only English, and we have 938 left

keyword = "questrade OR questwealth "
tweetCriteria = got.manager.TweetCriteria().setQuerySearch(keyword)\
                                           .setSince("2020-03-16")\
                                           .setUntil("2020-03-22")\
                                           .setLang('en')\
                                           .setMaxTweets(1000)
                                           
## only show the first twitter                                          
##  tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]
## print(tweet.text)

del tweets

tweets = got.manager.TweetManager.getTweets(tweetCriteria)

## show twitters
for tweet in tweets:
    print(tweet.text + '\n');





##############################################################################
##  try to  conver the results to a dataframe or a list of string
##  https://medium.com/@robbiegeoghegan/download-twitter-data-with-10-lines-of-code-42eb2ba1ab0f
## convert the tweet to a dataframe that we could analyze
del tweet_df,df,a

tweet_df = pd.DataFrame({'got_criteria':got.manager.TweetManager.getTweets(tweetCriteria)})

def get_twitter_info():
    tweet_df["tweet_text"] = tweet_df["got_criteria"].apply(lambda x: x.text)
    tweet_df["username"] = tweet_df["got_criteria"].apply(lambda x: x.username)
    tweet_df["date"] = tweet_df["got_criteria"].apply(lambda x: x.date)
    tweet_df["hashtags"] = tweet_df["got_criteria"].apply(lambda x: x.hashtags)
    tweet_df["link"] = tweet_df["got_criteria"].apply(lambda x: x.permalink)
    tweet_df["favorites"] = tweet_df["got_criteria"].apply(lambda x: x.favorites)
    tweet_df["replies"] = tweet_df["got_criteria"].apply(lambda x: x.replies)
    tweet_df["retweets"] = tweet_df["got_criteria"].apply(lambda x: x.retweets)
    
     
## run your function to get results    
get_twitter_info()


## remove the user name is "questrade", then 874 out of 938 left
tweet_df = tweet_df[tweet_df.username != "Questrade"]


## now apply the sentimental analysis to these txt
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



tweet_df['pos_score'] = tweet_df.tweet_text.apply(get_sentiment, k='pos')
tweet_df['neg_score'] = tweet_df.tweet_text.apply(get_sentiment, k='neg')
tweet_df['neu_score'] = tweet_df.tweet_text.apply(get_sentiment, k='neu')
tweet_df['compound']  = tweet_df.tweet_text.apply(get_sentiment, k='compound')


## quickly get average sentimental
## mean compound in 2020-03-16 to 2020-03-22 is  0.00589, very nerual
## mean compound in 2020-03-01 to 2020-03-07 is  0.148, positive
## after remove questrade comments, right now it's -0.2589
np.mean(tweet_df['compound'])




## export your data
tweet_df.to_csv(r"C:\Users\012790\Desktop\survey\tweet_mar16_22.csv",sep=",")


help(got.manager.TweetCriteria())


##################################################
## end of day 2020-03-23 things need to be done:
## 1. excluded some userid   
## 2. when turn to csv file, some punctuation will crash



















