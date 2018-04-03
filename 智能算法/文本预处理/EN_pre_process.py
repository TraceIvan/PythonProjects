import numpy as np
import random
import re
from nltk.corpus import stopwords
import chardet
import base64 as beautifulsoup

class English_text_preprocessing(object):
    def __init__(self):
        pass

    def file2StrByChardet(self,filename):
        '''将文件转化为包含文本的字符串,采用chardet解码'''
        with open(filename,'rb') as fr:
            txtstr=fr.read()
            encode_type = chardet.detect(txtstr)
            txtstr = txtstr.decode(encode_type['encoding'])  # 进行相应解码，赋给原标识符（变量）
        return txtstr

    def file2StrByUTF8(self,filename):
        '''将文件转化为包含文本的字符串，用utf8解码'''
        with open(filename,'rb') as fr:
            txtstr=fr.read().decode('utf8')
        return txtstr

    def file2StrByGBK(self,filename):
        '''将文件转化为包含文本的字符串，用gbk解码'''
        with open(filename,'rb') as fr:
            txtstr=fr.read().decode('gbk')
        return txtstr

    def removeURL(self,txtStr):
        '''去除文本字符串中的URL'''
        URL_pattern = re.compile('[a-zA-z]+://[^\s]*')
        URL_pattern_2=re.compile('^((https|http|ftp|rtsp|mms)?://)[^\s]+')
        txtStr=URL_pattern.sub(' ',txtStr)
        return txtStr

    def removeNumbers(self,wordList):
        '''去除单词表中的纯数字字符串'''
        Numbers_pattern = re.compile('\d+')
        return [word for word in wordList if not Numbers_pattern.match(word)]

    def removeStopWord(self,wordList):
        '''去除单词表里的停用词，
            需要引用模块：from nltk.corpus import stopwords
        '''
        cachedStopWords = stopwords.words("english")
        return [word for word in wordList if word not in cachedStopWords]

    def change2Lower_limitLength(self,wordList,limit_length=2):
        '''将单词表中的单词统一小写，并去除长度<=limitLength的单词'''
        return [word.lower() for word in wordList if len(word)>limit_length]

    def txt2list(self,txtStr):
        '''将文本转换为单词表'''
        wordList = re.split('\W+', txtStr)
        return wordList
    def solve_txt(self,txtStr):
        '''输入英文文本，输出预处理后的单词表'''
        # 去除URL
        txtStr = self.removeURL(txtStr)
        # 文本转换为单词表
        wordList = self.txt2list(txtStr)
        # 去除数字
        wordList = self.removeNumbers(wordList)
        # 去除停用词
        wordList = self.removeStopWord(wordList)
        # 小写化，同时去除长度小于2的单词
        wordList = self.change2Lower_limitLength(wordList, 2)
        return wordList

    def solve_file(self,filename):
        '''输入英文文本文件名（完整路径），输出预处理后的单词表'''
        txtStr=self.file2StrByChardet(filename)
        #去除URL
        txtStr=self.removeURL(txtStr)
        #文本转换为单词表
        wordList=self.txt2list(txtStr)
        #去除数字
        wordList=self.removeNumbers(wordList)
        #去除停用词
        wordList=self.removeStopWord(wordList)
        #小写化，同时去除长度小于2的单词
        wordList=self.change2Lower_limitLength(wordList,2)
        return wordList

def textParse(bigString):
    #encode_type = chardet.detect(bigString)
    #bigString = bigString.decode(encode_type['encoding'])  # 进行相应解码，赋给原标识符（变量）
    URL_pattern=re.compile('[a-zA-z]+://[^\s]*') #re.compile(ur'^((https|http|ftp|rtsp|mms)?://)[^\s]+')
    bigString=URL_pattern.sub(' ',bigString)#去除URL
    listOfTokens=re.split('\W+',bigString)
    #listOfTokens=[tok.lower() for tok in listOfTokens if len(tok) > 2]
    removeNumbers = re.compile('\D+')
    listOfTokens=[tok for tok in listOfTokens if removeNumbers.match(tok)]#去除所有数字字符串
    #去除停用词、小写化、去除短词
    cachedStopWords = stopwords.words("english")
    cleanWords = [w.lower() for w in listOfTokens if w.lower() not in cachedStopWords and 3 <= len(w)]

    return cleanWords