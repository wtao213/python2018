



#---------------------------------------

# NOTES

# Program dependency
    # _reddit_data_prep_mar (generated march data df_mar)
    # _reddit_scraper_v2 (scrapes new data)

# Versions
# _reddit_mortgg_scraper: old scraping method
# _reddit_mortgg_scraper_v2: new scraping method, dealing with data of march & april 2021
# _reddit_mortgg_scraper_v3: new method, dealing with data since march 2021 onward

#---------------------------------------




#----- 1.Import Data

#---- import april data

#--- df1_posts

# do this when df1_posts contains all june data

# import previous scrape
df1_posts = pd.read_pickle("C:\\Users\\013125\\.venv\\Scripts\\df1_posts_20210511.pkl")


# filter data after april 1st for df1
df1_posts_since_jun = df1_posts[df1_posts["created_dttm"].dt.month == 6]

# may.11 391
# may.18 420
# jun.8 96



#--- df2_posts
# df2 needs a bit of stacking since each scrape contains less than 250 posts
# it's about 9-10 days of data normally so scrape weekly is good enough
# but if there are more posts,
# the number of days containing in each scrape diminishes
# in which case we might need to scrape more frequently

# df1_posts is included in df2_posts, given the same time period


# import previous scrapes
# df2_posts_20210407 = pd.read_pickle("C:\\Users\\013125\\.venv\\Scripts\\df2_posts_20210407.pkl")






#----- 2.Data Prep

# keep only data after june 1st
df2_posts_since_jun = df2_posts[df2_posts["created_dttm"].dt.month == 6]

# may.11 1388
# may.18 1637
# jun.8 203



# sort by post id, then post date
#df2_posts_since_jun = df2_posts_since_jun.sort_values(['id', 'created'], ascending = [True, True]).reset_index(drop=True)


# deduplicate, keep the first entry (doesn't matter first or last)
#df2_posts_since_jun = df2_posts_since_jun.drop_duplicates(subset=['id'], keep='first')

# may.11 935
# may.18 1043
# jun.8 203


# df2_posts_since_apr.to_csv('df2_posts_since_apr.csv')



# stack df1_posts and df2_posts
df_since_jun = pd.concat([df2_posts_since_jun, df2_posts_since_jun])

# may.11 1326
# may.18 1463
# jun.8 406


# deduplicate, keep the first entry (doesn't matter first or last, there will be variation based when the data is pulled)
df_since_jun = df_since_jun.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)

# may.11 935
# may.18 1043
# jun.8 203


# df_since_apr.to_csv('df_since_apr.csv')




#---- import mar-may data
df_mar = pd.read_pickle(".\\data\\Reddit_data\\Reddit_202103_cleaned.pkl") # 1083
df_apr = pd.read_pickle(".\\data\\Reddit_data\\Reddit_202104.pkl") # 730
df_may = pd.read_pickle(".\\data\\Reddit_data\\Reddit_202105.pkl") # 495
df_jun = pd.read_pickle(".\\data\\Reddit_data\\Reddit_202106.pkl") # 561



# datasets march-may have 2 dates, delete created_dttm

df_3_5 = pd.concat([df_mar, df_apr, df_may])
del df_3_5['created_dttm']


# rename created_dttm2 and created_dt2 to created_dttm and created_dt
df_3_5.rename(columns={'created_dttm2': 'created_dttm', 'created_dt2': 'created_dt'}, inplace=True)


# stack jun data
df_since_mar = pd.concat([df_3_5, df_jun])


# create various date format

def create_date(df, dttm):
    # date
    df['created_dt'] = dttm.dt.date

    # month and year
    df['created_mth'] = dttm.dt.month
    df['created_yr'] = dttm.dt.year

    # month + year
    df['created_my'] = df['created_yr'].astype("str") + '-' + df['created_mth'].astype("str")
    df['created_my'] = df['created_my'].astype("str")

#create_date(df=df_tw_2021, dttm=df_tw_2021['date'])
create_date(df=df_since_mar, dttm=df_since_mar['created_dttm'])



# mar-may: 2308
# jun.8 2511
# mar-jun: 2869



#----- 3.Search for mortgage-related posts


# combine title and body
df_since_mar['text'] = df_since_mar[['title', 'body']].apply(lambda x: ' '.join(x), axis=1)

# clean text, lower case
df_since_mar['text_cleaned'] = df_since_mar.text.apply(round1)


df_since_mar.to_csv('.//data//df_since_mar_20210630.csv')





#===== Search 1
# search for term "mortgage" in df
list_ = ['mortgage'] # this contains 'mortgages', 'xxxmortgage', "Mortgage", "MORTGAGE" (not case sensitive)
mask = df_since_mar.text_cleaned.apply(lambda x: any(item for item in list_ if item in x))
#print(len(mask))



# filter all rows that contain those key words
df_mort_since_mar = df_since_mar[mask]

# apr.12 14
# may.11 22
# may.18 
# until jun.30 on jul.13: 32

df_mort_since_mar.to_csv('.//data//df_mort_since_mar_20210630.csv')





#===== Search 2
# search for term "QuestMortgage" in entire reddit
# use "questmortgage" as search term; "QuestMortgage" pull out posts that don't contain the term for unknown reason

search = reddit.subreddit("all").search("questmortgage", sort="new", time_filter="all", limit=None)

# score and convert reddit search object to dataframe

df_qm = scrape_reddit_posts(search)

# may.11 4
# jun.8 4
# jul.13 4


df_qm["created_dttm"] = df_qm['created'].apply(get_datetime)
df_qm["created_dt"] = df_qm['created'].apply(get_date)


# combine title and body
df_qm['text'] = df_qm[['title', 'body']].apply(lambda x: ' '.join(x), axis=1)

# clean text, lower case
df_qm['text_cleaned'] = df_qm.text.apply(round1)


df_qm.to_csv('.//data//df_qm_20210713.csv')



#--- Stack and dedup

df_mort_all = pd.concat([df_mort_since_mar, df_qm])

len(df_mort_all)
# jun.8 32 mortgage-related posts
# jul.13 36

# check
df_mort_all.to_csv('.//data//df_mort_all_20210713.csv')


#-- dedup

#df_mort_all = df_mort_all.sort_values(['id', 'created'], ascending = [True, True]).reset_index(drop=True)
df_mort_all_dd = df_mort_all.drop_duplicates(subset=['id'], keep='first')

len(df_mort_all_dd)
# jun.8 28 mortgage-related posts
# jul.32

df_mort_all_dd.to_csv('.//data//df_mort_all_20210713.csv')




#===== Search 3 (Optional)
# search for term "Quest Mortgage" in entire reddit
# this will search for any posts containing 'quest' AND containing 'mortgage'

search2 = reddit.subreddit("all").search("Quest Mortgage", sort="new", time_filter="all", limit=None)

df_qm2 = scrape_reddit_posts(search2)

df_qm2.to_csv('df_qm2_20210511.csv')


