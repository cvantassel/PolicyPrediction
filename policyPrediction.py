import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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
print(df["user_location"].value_counts().index[0:15])

# Get Sentiment from Vader
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
df["Score"] = df["text_clean"].apply(lambda x: sid.polarity_scores(x)) 
print(df["Score"])