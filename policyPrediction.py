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

#Number of random locations listed
print("No. Of Original Unique Locations:",df["user_location"].nunique())
#dropping all null user locations
df.dropna(subset=['user_location'], inplace=True)


#making locations such as New York fall under the United States catagory
df.loc[df['user_location'].str.contains('India|Delhi', case=False)] = "India"
df.loc[df['user_location'].str.contains('USA|United States|DC|NY|CA|MA|MD|GA|TX|PA|WA|York|Texas|Los|Florida', case=False)] = "United States"
df.loc[df['user_location'].str.contains('Toronto', case=False)] = "Canada"
df.loc[df['user_location'].str.contains('England|UK|London', case=False)] = "United Kingdom"
df.loc[df['user_location'].str.contains('Nigeria', case=False)] = "Nigeria"
df.loc[df['user_location'].str.contains('China|Kong', case=False)] = "China"
df.loc[df['user_location'].str.contains('Australia', case=False)] = "Australia"
df.loc[df['user_location'].str.contains('Ireland', case=False)] = "Ireland"
df.loc[df['user_location'].str.contains('Switzerland', case=False)] = "Switzerland"
df.loc[df['user_location'].str.contains('Worldwide|WORLD|Earth|Global|Text|Europe', case=False)] = "USELESS"

#Number of locations after filtering out the excess
print("No. Of Unique Locations:",df["user_location"].nunique())

#graph of the top locations with the number of tweets in each respective country
plt.figure(figsize=(10,12))
sns.barplot(df["user_location"].value_counts().values[0:20],
            df["user_location"].value_counts().index[0:20]);
plt.title("Top 15 Location")
plt.xlabel("No. of tweets")
plt.ylabel("Location")
plt.show()