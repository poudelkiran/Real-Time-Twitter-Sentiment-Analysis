
import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment as s

#Inorder to obtain keys and token, one should need to create an APP in twitter.
# Authentication 
consumer_key= 'Your consumer_key'
consumer_secret= 'Your consumer_secret'
access_token='your access_token'
access_token_secret='your access_token_secret'


#Get Real Time Tweets from the twitter.
class listener(StreamListener):
    def on_data(self, data):
        all_data = json.loads(data)
        tweet = all_data["text"]
#Find the Sentiment of the tweet by calling sentiment function inside sentiment file.
        sentiment_value, confidence = s.sentiment(tweet)
        if confidence*100 >= 80:
            output = open("Output/twitter-out.txt","a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()
        return True
    def on_error(self, status):
        print(status)

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
twitterStream = Stream(auth, listener())

#Enter the Keyword for which you wish to analyze the tweets.
key = input("Enter the Keyword for real time sentiment analysis? ")
twitterStream.filter(track=[key])

