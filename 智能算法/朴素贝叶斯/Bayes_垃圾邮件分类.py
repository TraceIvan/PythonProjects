import Bayes_简单侮辱性言论分类 as basic_bayes
import random
import re
import numpy as np
from nltk.corpus import stopwords
import chardet   #需要导入这个模块，检测编码格式
#文本解析
#去掉少于2个字符的字符串，并将所有字符串转换为小写
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
#垃圾邮件分类
def spamTest():
    docList=[]
    classList=[]
    fullText=[]
    #导入并解析文本文件
    for i in range(1,26):
        with open('email/spam/%d.txt'%i,'rb') as fr:
            wordList=textParse(fr.read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        with open('email/ham/%d.txt'%i,'rb') as fr:
            wordList=textParse(fr.read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList=basic_bayes.createVocabList(docList)
    #构建测试集和训练集
    trainingSet=list(range(50))
    testSet=[]
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat=[]
    trainClasses=[]
    #循环遍历训练集，构建词向量
    for docIndex in trainingSet:
        trainMat.append(basic_bayes.setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=basic_bayes.trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount=0
    #对测试集分类
    for docIndex in testSet:
        wordVector=basic_bayes.setOfWords2Vec(vocabList,docList[docIndex])
        classified_result=basic_bayes.classifyNB(np.array(wordVector),p0V,p1V,pSpam)
        if classified_result!=classList[docIndex]:
            errorCount+=1
            print('the predicted answer is %d,the true is %d.'%(classified_result,classList[docIndex]))
    print('the error rate is : %f.'%(float(errorCount)/len(testSet)))

if __name__=='__main__':
    spamTest()