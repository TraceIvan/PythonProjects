import json
import sys
import codecs
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
from os import path
import tweepy

if __name__=='__main__':
    d=path.dirname("__file__")
    with open('filtered_tweets.txt','r',encoding='utf8') as rf:
        words=rf.read()
        wordcloud=WordCloud(font_path="C:\Windows\Fonts\simhei.ttf",stopwords=STOPWORDS,background_color='#222222',
                        width=1000,height=800).generate(words)
        plt.figure(1)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.show()