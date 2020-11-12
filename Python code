
# import tweepy
import codecs
import numpy as np 
import pandas as pd
from textblob import TextBlob
import re 
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud




#variables that contain the user credentials to access Twitter API.

CONSUMER_KEY = "xyz"
CONSUMER_SECRET = "xyz"
ACCESS_TOKEN = "xyz"
ACCESS_TOKEN_SECRET = "xyz"
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)




#authorisation 
api = tweepy.API(auth,wait_on_rate_limit = True, wait_on_rate_limit_notify=True)


#defined function for tweets collection
def get_tweet(tweet, default_value=None):
    data = {}
    data['id'] = tweet.user.id if tweet.id else default_value
    data['full_text'] = tweet.full_text if tweet.full_text else default_value
    data['created_at'] = tweet.created_at if tweet.created_at else default_value
    
    return data    


#function called
file1 = codecs.open("try.text","w",encoding="utf-8")
q = "elonmusk OR SpaceX OR SolarCity OR Tesla OR Starlink OR Rockets -filter:retweets"
posts = [get_tweet(tweet) for tweet in 
         tweepy.Cursor(api.search, q = q, 
                       lang="en", 
                       since = "2020-05-03",
                       until = "2020-05-09",
                       tweet_mode="extended").items()]
df = pd.DataFrame(posts)





print("Summary of the basic information about this DataFrame and its data:")
print(df.info())


df[0:5]





#DATA PRE-PROCESSING 

#defined function for punctuation marks removal
def preproces(line):

    line = re.sub(r'\.+',"", line)
    line = re.sub('@+',"", line)
    line = re.sub(r'\!+',"", line)
    line = re.sub('https?:\/\/\S+', '', line)
    line = re.sub(r'\#+',"", line)
    line = re.sub(r'\,+',"", line)
    line = re.sub(r'\’+',"", line)
    line = re.sub(r'\_+',"", line)
    line = re.sub(r'\-+',"", line)
    return line


#function called
df["full_text"] = df['full_text'].apply(preproces)


#changing strings to lowercase
df['full_text'] = df['full_text'].str.lower()


#changing datetime to date format
df['created_at'] = pd.to_datetime(df['created_at']).dt.date


#Removing stopwords
esw = stopwords.words('english')
esw.append("would")
esw.append("via")

df['full_text'] = df["full_text"].apply(lambda remove:' '.join([words for words in remove.split() if words not in (esw)]))



df[0:5]





#DATA ANALYSIS

#Counting top 10 most frequent words
from collections import Counter
top10 = Counter(" ".join(df["full_text"]).split()).most_common(10)


#Converting results into dataframe
top = pd.DataFrame(top10,columns = ['WORD','FREQUENCY'])
top.index.name = 'TOP 10 WORDS'
top.index += 1
top

ax = top.plot(x="WORD", y=["FREQUENCY"], kind="bar", figsize=(18,10), fontsize=16)
ax.set_xlabel("WORD")
ax.set_ylabel("FREQUENCY")


from wordcloud import WordCloud, STOPWORDS
import matplotlib
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (16.0, 9.0)



#WORD CLOUD

# Reading the script
script = ' '.join([twts for twts in df['full_text']])
# Create WordCloud Object
wc = WordCloud(background_color="white", 
               width=1600, height=900, colormap=matplotlib.cm.inferno)
# Generate WordCloud
wc.generate(script)
# Show the WordCloud
plt.figure()
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")


x = df.groupby('created_at')['id'].nunique()
x

x = df.groupby('created_at')['full_text'].nunique()
x



#PLOTTING

fig, ax = plt.subplots(figsize=(14,9))
df.groupby('created_at')['id'].nunique().plot(kind ='bar',
                                              color=['red', 'orange', 'darkblue', 'darkgreen', 'pink', 'brown'])
plt.title("NUMBER OF TWEETS POSTED PER DAY")
plt.xlabel("DATE")
plt.ylabel("NUMBER OF TWEETS")



fig, ax = plt.subplots(figsize=(14,9))
df.groupby('created_at')['full_text'].nunique().plot(kind ='bar',
                                              color=['red', 'orange', 'darkblue', 'darkgreen', 'pink', 'brown'])
plt.title("NUMBER OF UNIQUE USERS PER DAY")
plt.xlabel("DATE")
plt.ylabel("NUMBER OF UNIQUE USERS")


#Top 10 most active users over the period of one week
Top10 = df.groupby('screen_name')['id'].count()
Top10.nlargest(10).plot.bar(figsize=(14,9), colormap='Set1')
plt.title=('TOP 10 MOST ACTIVE USERS')
plt.xlabel("USERNAME")
plt.ylabel("NUMBER OF TWEETS")
print(Top10.nlargest(10))




#SENTIMENT MODELLING

#Sentiment analysis - counting number of tweets with sentiments

positive = 0
negative = 0
neutral = 0

for tweet in df["full_text"]:
    sen = TextBlob(tweet)
    if(sen.sentiment.polarity==0):
        neutral = neutral +1
    elif(sen.sentiment.polarity > 0):
        positive = positive+1
    elif (sen.sentiment.polarity < 0):
        negative = negative +1

#Result
print(positive)
print(negative)
print(neutral) 



#Sentiment analysis using NaiveBayesAnalyzer

from textblob.sentiments import NaiveBayesAnalyzer

prebuilt_classifier = NaiveBayesAnalyzer()
for tweet in df['full_text']:
    value = TextBlob(tweet,analyzer=prebuilt_classifier)

#Result
value.sentiment




#Sentiment analysis -  counting the number of noun phrases
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from textblob.np_extractors import ConllExtractor

extractor = ConllExtractor()
prebuilt_analyzer = NaiveBayesAnalyzer()

positive = 0
negative = 0
neutral = 0

np={}
for line in df["full_text"]:
    lc_line = line.lower()
    
    if "elon" in lc_line and "musk" in lc_line:
        
        tweet = TextBlob(lc_line, analyzer=prebuilt_analyzer, np_extractor=extractor)
        
        for n in tweet.noun_phrases:
            if n in np:
                np[n] += 1
            else:
                np[n] = 1

        if tweet.sentiment.p_pos >= 0.7:
            positive +=1
        elif tweet.sentiment.p_neg >= 0.7:
            negative +=1
        else:
            neutral +=1



#Results
print(positive)
print(negative)
print(neutral)


for w in sorted(np, key=np.get, reverse=True)[0:5]:
    print(w, np[w])






