from nltk.corpus import stopwords
cachedStopWords = stopwords.words("english")
import feedparser
from os import path
import re
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签(黑体)
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
from bs4 import BeautifulSoup
import numpy as np



d=path.dirname("__file__")
mystopwords=cachedStopWords
#rss信息源列表
feedList=['http://www.engadget.com/rss.xml']
'''
'http://www.techcrunch.com/rssfeeds/',
          'http://www.computerweekly.com/rss',
          'http://feeds.twit.tv/tnt.xml',
          'https://www.apple.com/pr/feeds/pr.rss',
          'https://news.google.com/?output=rss',
          'http://www.forbes.com/technology/feed/',
          'http://rss.nytimes.com/services/xml/rss/nyt/Technology.xml',
          'http://www.nytimes.com/roomfordebate/topics/technology.rss',
          'http://feeds.webservice.techradar.com/us/rss',
          'http://feeds.webservice.techradar.com/us/rss/reviews',
          'http://feeds.webservice.techradar.com/us/rss/news/software',
          'http://www.cnet.com/rss/',
          'http://feed.feedburner.com/ibm-big-data-hub?format=xml',
          'http://feed.feedburner.com/ResearchDiscussions-DataScienceCentral?format=xml',
          'http://feed.feedburner.com/BdnDailyPressReleasesDiscussions-BigDataNews?format=xml',
          'http://feed.feedburner.com/ibm-big-data-hub-galleries?format=xml',
          'http://feed.feedburner.com/PlanetBigData?format=xml',
          'http://rss.cnn.com/rss/cnn_tech.rss',
          'http://news.yahoo.com/rss/tech',
          'http://slashot.org/slashdot.rdf',
          'http://bbc.com/news/technology/'
'''
def extractPlainText(ht):
    soup=BeautifulSoup(ht,'lxml')
    plaintxt=''
    s=0
    for char in ht:
        if char =='<':
            s=1
        elif char=='>':
            s=0
            plaintxt+=' '
        elif s==0:
            plaintxt+=char
    return plaintxt

def separatewords(text):
    splitter=re.compile('\W+')
    return [s.lower() for s in splitter.split(text) if len(s)>3]

def combineWordsFromFeed(filename):
    with open(filename,'w',encoding='utf8') as wfile:
        for feed in feedList:
            print('Parsing '+feed)
            fp=feedparser.parse(feed)
            txt=''
            if len(fp.entries) ==0:
                continue
            for e in fp.entries:
                if 'describtion' in e.keys():
                    txt=e.title.encode('utf8')+extractPlainText(e.describtion.encode('utf8'))
                elif 'summary' in e.keys():
                    txt = e.title.encode('utf8') + extractPlainText(e.summary.encode('utf8'))
            words=separatewords(txt)

            for word in words:
                if word.isdigit()==False and word not in mystopwords:
                    wfile.write(word)
                    wfile.write(" ")
                wfile.write("\n")
    wfile.close()
    return

if __name__=='__main__':
    combineWordsFromFeed('wordcloudInput_FromFeeds.txt')
    d=path.dirname("__file__")
    with open(path.join(d,'wordcloudInput_FromFeeds.txt')) as rf:
        text=rf.read()

        wordcloud=WordCloud(font_path='C:\Windows\Fonts\simhei.ttf',stopwords=STOPWORDS,background_color='#222222',
                            width=1000,height=800).generate(text)

        plt.figure(1)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.show()