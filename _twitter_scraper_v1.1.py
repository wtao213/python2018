

#---------------------------------------

# NOTES

# This script is used to scrape twitter data
# and search for questmortgage-related tweets

#---------------------------------------



import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


#https://github.com/scalto/snscrape-by-location/blob/main/snscrape_by_location_tutorial.ipynb
#pip install --upgrade git+https://github.com/JustAnotherArchivist/snscrape@master
#pip3 install git+https://github.com/JustAnotherArchivist/snscrape.git

# our search term, using syntax for Twitter's Advanced Search
#search = '"questrade"'

# the scraped tweets, this is a generator
scraped_tweets = sntwitter.TwitterSearchScraper('questrade + since:2021-06-01 until:2021-07-01').get_items()


# slicing the generator to keep only the first 100 tweets
#sliced_scraped_tweets = itertools.islice(scraped_tweets, 100)


# convert to a DataFrame and keep only relevant columns
df_tw = pd.DataFrame(scraped_tweets) #[['url', 'date', 'content', 'id', 'username', 'outlinks', 'outlinksss', 'tcooutlinks', 'tcooutlinksss']]
# may1 2021-may31 2021: 671
# jun1 2021-jun31 2021: 892


#--- dev version
#df_tw = pd.DataFrame(scraped_tweets)[['url', 'date', 'content', 'id', 'username', 'tcooutlinks']]
#df_tw.columns


#--- export

df_tw.to_pickle('.\\data\\Twitter_data\\df_tw_jun2021.pkl')
df_tw.to_csv('.\\data\\Twitter_data\\df_tw_jun2021.csv')







### set display ###
#np.set_printoptions(threshold=np.nan)
pd.set_option('display.max_colwidth', -1)






#------------------------
# mortgage
#------------------------

##### Search 1: 

df_tw['content_cleaned'] = df_tw.content.apply(round1)

# search for term "mortgage" in df
list_ = ['mortgage']
mask = df_tw.content_cleaned.apply(lambda x: any(item for item in list_ if item in x))
#print(len(mask))

# examine all rows that contain those key words
df_tw_mort = df_tw[mask]

df_tw_mort
# may.11 13
# may.18 16


#print(len(df_keyword))
df_tw_mort.iloc[7, 2]



df_tw_mort.to_csv('df_tw_mort_20210517.csv')


##### Search 2: 

# search for term "QuestMortgage" in entire twitter
scraped_tweets_qm = sntwitter.TwitterSearchScraper('questmortgage + since:2021-03-8 until:2021-04-30').get_items()

df_tw_qm = pd.DataFrame(scraped_tweets_qm)
# may.11 6


#----- stack two mortgage search result

df_tw_mort_all = pd.concat([df_tw_mort, df_tw_qm])

# dedup
df_tw_mort_all = df_tw_mort_all.drop_duplicates(subset=['url'], keep='first')
# may.11 15

df_tw_mort_all.to_csv('df_tw_mort_all_20210430.csv')


