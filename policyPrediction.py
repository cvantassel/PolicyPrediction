import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import seaborn as sns
plt.style.use("fivethirtyeight")
sns.set()
sns.palplot(sns.color_palette())



df = pd.read_csv("covid19_tweets.csv")

# exTweet = "strive to promote Truth with Integrity. https://t.co/pOqdhk6Mhg"
def clean_tweet(text):
    ''' Remove links and symbols, make text lowercase'''
    text = text.lower()
    text = re.sub('http\S+', '', text)
    text = re.sub('[^0-9a-zA-Z ]+', '', text)
    return text

df['text_clean'] = df['text'].apply(str).apply(lambda x: clean_tweet(x))

# Print locations
print(df["user_location"].value_counts().index[0:500])

# Get Sentiment from Vader
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
df["Score"] = df["text_clean"].apply(lambda x: sid.polarity_scores(x)) 
print(df["Score"])

print("No. Of Original Unique Locations:",df["user_location"].nunique())

df.dropna(subset=['user_location'], inplace=True)

#if (df['user_location'].__contains__('India')):
#    df['user_location'] = 'India'


df.loc[df['user_location'].str.contains('India', case=False)] = "India"
df.loc[df['user_location'].str.contains('Delhi', case=False)] = "India"

df.loc[df['user_location'].str.contains('USA', case=False)] = "United States"
df.loc[df['user_location'].str.contains('United States', case=False)] = "United States"
df.loc[df['user_location'].str.contains('DC', case=False)] = "United States"
df.loc[df['user_location'].str.contains('NY', case=False)] = "United States"
df.loc[df['user_location'].str.contains('CA', case=False)] = "United States"
df.loc[df['user_location'].str.contains('MA', case=False)] = "United States"
df.loc[df['user_location'].str.contains('MD', case=False)] = "United States"
df.loc[df['user_location'].str.contains('GA', case=False)] = "United States"
df.loc[df['user_location'].str.contains('TX', case=False)] = "United States"
df.loc[df['user_location'].str.contains('PA', case=False)] = "United States"
df.loc[df['user_location'].str.contains('WA', case=False)] = "United States"
df.loc[df['user_location'].str.contains('York', case=False)] = "United States"
df.loc[df['user_location'].str.contains('Texas', case=False)] = "United States"
df.loc[df['user_location'].str.contains('Los', case=False)] = "United States"
df.loc[df['user_location'].str.contains('Florida', case=False)] = "United States"

df.loc[df['user_location'].str.contains('Toronto', case=False)] = "Canada"

df.loc[df['user_location'].str.contains('England', case=False)] = "United Kingdom"
df.loc[df['user_location'].str.contains('UK', case=False)] = "United Kingdom"
df.loc[df['user_location'].str.contains('London', case=False)] = "United Kingdom"

df.loc[df['user_location'].str.contains('Nigeria', case=False)] = "Nigeria"

df.loc[df['user_location'].str.contains('China', case=False)] = "China"
df.loc[df['user_location'].str.contains('Kong', case=False)] = "China"

df.loc[df['user_location'].str.contains('Australia', case=False)] = "Australia"

df.loc[df['user_location'].str.contains('Ireland', case=False)] = "Ireland"
df.loc[df['user_location'].str.contains('Switzerland', case=False)] = "Switzerland"

df.loc[df['user_location'].str.contains('Worldwide', case=False)] = "USELESS"
df.loc[df['user_location'].str.contains('WORLDWIDE', case=False)] = "USELESS"
df.loc[df['user_location'].str.contains('Earth', case=False)] = "USELESS"
df.loc[df['user_location'].str.contains('Global', case=False)] = "USELESS"
df.loc[df['user_location'].str.contains('Text', case=False)] = "USELESS"
df.loc[df['user_location'].str.contains('Europe', case=False)] = "USELESS"



print("No. Of Unique Locations:",df["user_location"].nunique())



plt.figure(figsize=(10,12))
sns.barplot(df["user_location"].value_counts().values[0:20],
            df["user_location"].value_counts().index[0:20]);
plt.title("Top 15 Location")
plt.xlabel("No. of tweets")
plt.ylabel("Location")
plt.show()