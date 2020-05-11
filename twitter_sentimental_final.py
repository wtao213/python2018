# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:41:28 2020

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
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
import GetOldTweets3 as got
import csv




####################################################################################
## 2020-02-19 is the max dow jones day from that day till now, there are 1627 return tweets
## 


## might search for French in the future, location
## 03-23 not included

keyword = "questrade OR questwealth"
tweetCriteria = got.manager.TweetCriteria().setQuerySearch(keyword)\
                                           .setSince("2020-03-26")\
                                           .setUntil("2020-03-31")\
                                           .setLang('en')\
                                           .setMaxTweets(2000)
                                           
                                           
                                           
##############################################################################
##  try to  conver the results to a dataframe or a list of string
##  https://medium.com/@robbiegeoghegan/download-twitter-data-with-10-lines-of-code-42eb2ba1ab0f
## convert the tweet to a dataframe that we could analyze
del tweet_df

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
## remove user, 1487 out of 1627 left
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

tweet_df['date_o'] = pd.to_datetime(tweet_df.date)
tweet_df['date_o'] = tweet_df['date_o'].dt.date

## quickly get average sentimental
## mean compound in 2020-03-16 to 2020-03-22 is  0.00589, very nerual
## mean compound in 2020-03-01 to 2020-03-07 is  0.148, positive
## after remove questrade comments, right now it's -0.0291
## only march 23,24 two days there are 1488 twetts and overall -0.064 more negative
np.mean(tweet_df['compound'])




## export your data
## because of the special characters, copy paste directly from dataframe is most effeicient way
tweet_df.to_csv(r"C:\Users\012790\Desktop\survey\tweet_feb19_mar23.csv",sep=",",index= False)




