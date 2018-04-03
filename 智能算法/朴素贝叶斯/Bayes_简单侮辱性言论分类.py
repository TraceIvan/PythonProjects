import numpy as np

#创建实验样本
def loadDataSet():
    postingList=[['my','dog','has','flea','problems','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how','to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]#1代表侮辱性文字，0代表正常言论
    return postingList,classVec
#创建一个包含在所有文档中出现的不重复词表
def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)#创建并集
    return list(vocabSet)
#词集模型，将每个词的出现与否作为一个特征
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print('the word: %s is not in my Vocabulary!'%word)
    return returnVec
#词袋模型，统计每个词出现的次数
def bagOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec
#朴素贝叶斯分类器训练函数
#参数：文档矩阵；文档类别标签向量
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)#文档属于侮辱性的概率P(1)
    #初始化
    #当计算P(w0,w1,w2...,wn|ci)=P(w0|ci)*P(w1|ci)*...*P(wn|ci)时，其中若有一个为0，则结果为0
    #为降低这种影响，可以将所有出现数目初始化为1，并将分母初始化为2(类别数目)-->Laplace smoothing
    p0Num=np.ones(numWords)#p0Num=np.zeros(numWords)
    p1Num=np.ones(numWords)#p1Num=np.zeros(numWords)
    p0Denom=2.0#p0Denom=0.0
    p1Denom=2.0#p1Denom=0.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    # 对每一个元素除以该类别中的总词数P(w0|c1),P(w1|c1),P(w2|c1)...P(Wn|C1)
    p1Vect = np.log(p1Num / p1Denom)#p1Vect=p1Num/p1Denom,使用log防止下溢出
    p0Vect=np.log(p0Num/p0Denom)#P(w0|c0),P(w1|c0),P(w2|c0)...P(Wn|C0)
    return p0Vect,p1Vect,pAbusive
#朴素贝叶斯分类函数
#参数：要分类的向量；P(w0|c0),P(w1|c0),P(w2|c0)...P(Wn|C0)；P(w0|c1),P(w1|c1),P(w2|c1)...P(Wn|C1)；P(1)
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+np.log(pClass1)#log(P(W|c1)*P(c1)/P(W))
    p0=sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses=loadDataSet()
    myvocalList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myvocalList,postinDoc))
    p0V,p1V,pAb=trainNB0(trainMat,listClasses)
    testEntry=['love','my','dalmation']
    thisDoc=np.array(setOfWords2Vec(myvocalList,testEntry))
    print(testEntry,'classified as : ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['garbage','stupid']
    thisDoc = np.array(setOfWords2Vec(myvocalList, testEntry))
    print(testEntry, 'classified as : ', classifyNB(thisDoc, p0V, p1V, pAb))

if __name__=='__main__':
    testingNB()#是否是侮辱性分类
