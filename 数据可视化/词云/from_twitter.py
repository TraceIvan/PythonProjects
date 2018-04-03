import json
import sys
import codecs
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
from os import path
import tweepy
counter=0
MAX_TWEETS=100#设置最多提取的tweet的消息数目
access_token="979004315573481473-6UBJbqrXa1o8ZVW91HBeQ7PyAJsGv1h"
access_token_secret="myarGIHoFpFO0YNZ9SH3jyTCxVuD4871cIxm7vAqqU2BX"
consumer_key="SADPG8Bs0Z8YsPIQpWsFLteOM"
consumer_secret="QDRH9IVtyFUzLE2X8y99mg9BzZiVnbmJ2VQnrpaPzKklfI9Vjf"
fp=codecs.open('filtered_tweets.txt','w',encoding='utf8')
#设置简单的流监听器
class CustomStreamListener(tweepy.StreamListener):
    def on_status(self,status):
        global counter
        fp.write(status.text)
        print('Tweet-count: '+str(counter))
        counter+=1
        if counter>=MAX_TWEETS:
            sys.exit()

    def on_error(self,status):
        print(status)

if __name__=='__main__':
    auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_token,access_token_secret)
    streaming_api=tweepy.streaming.Stream(auth,CustomStreamListener(),timeout=60)
    streaming_api.filter(track=['python program','statistics','data visualization','big data','machine learning'])#监听内容


