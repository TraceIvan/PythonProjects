import Bayes_简单侮辱性言论分类 as basic_bayes
import Bayes_垃圾邮件分类 as basic_bayes2
import random
import re
import operator
import feedparser
import numpy as np
#遍历词汇表中的每个词并统计其在文本中出现的次数
#返回出现次数从高到低的前30个单词
def calcMostFreq(vocabList,fullText):
    freqDict={}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq=sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]
#使用两个rss源作为参数
def localWords(feed1,feed0):
    docList=[]
    classList=[]
    fullText=[]
    minLen=min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList=basic_bayes2.textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=basic_bayes2.textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=basic_bayes.createVocabList(docList)
    #去除高频词
    top30Words=calcMostFreq(vocabList,fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
            
    trainingSet=list(range(2*minLen))
    testSet=[]
    for i in range(20):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(basic_bayes.bagOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=basic_bayes.trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector=basic_bayes.bagOfWords2Vec(vocabList,docList[docIndex])
        classified_re=basic_bayes.classifyNB(np.array(wordVector),p0V,p1V,pSpam)
        if classified_re!=classList[docIndex]:
            errorCount+=1
            print('the predicted answer is %d, the true is %d.'%(classified_re,classList[docIndex]))
    print('the error rate is %f.'%(float(errorCount)/len(testSet)))
    return vocabList,p0V,p1V,pSpam

def getTopWords(ny,sf):
    vocabList,p0V,p1V,pSpam=localWords(ny,sf)
    print(p0V)
    print(p1V)
    topNY=[]
    topSF=[]
    for i in range(len(p0V)):
        if p0V[i]>-5:topSF.append((vocabList[i],p0V[i]))
        if p1V[i]>-5:topNY.append((vocabList[i],p1V[i]))
    sortedSF=sorted(topSF,key=lambda pair:pair[1],reverse=True)
    print('SF*******SF:')
    for item in sortedSF:
        print(item[0])
    sortedNY=sorted(topNY,key=lambda  pair:pair[1],reverse=True)
    print('NY*******NY:')
    for item in sortedNY:
        print(item[0])
if __name__=='__main__':
    ny=feedparser.parse('http://newyork.craigslist.org/jjj/index.rss')
    print(ny)
    sf=feedparser.parse('http://sfbay.craigslist.org/jjj/index.rss')
    getTopWords(ny,sf)

