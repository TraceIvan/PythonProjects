import math
import numpy
from collections import Counter
from functools import reduce
import operator
import pickle
import matplotlib.pyplot as plt
#定义决策节点、叶节点和箭头格式
decisionNode=dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def loadData(fileName):
    dataArr=[]
    with open(fileName,'r',encoding='utf8') as fr:
        for line in fr.readlines():
            lineArr=line.strip().split(',')
            dataArr.append(list(map(float,lineArr[:-1])))
            dataArr[-1].append(lineArr[-1])
    return dataArr

#计算给定数据集的乡农熵(信息熵)
#熵越高，则表明混合的数据越多,纯度越低，有序程度越低，分类效果越差
def calcshannonEnt(dataSet):
    numEntries=len(dataSet)#计算数据集中实例的总数
    labelCounts={}
    #为所有可能分类创建字典，表示类别出现的次数
    for featVec in dataSet:
        currentLabel=featVec[-1]#假设最后一列为类标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries#发生概率
        shannonEnt-=prob*math.log(prob,2)
    return shannonEnt

#按照给定特征划分数据集
#参数：待划分的数据集；划分数据集的特征；需要返回特征的值
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最好的数据集划分方式
#要求：dataSet为列表元素组成的列表；所有列表元素具有相同的长度；每一个实例的最后一
# 个数据为当前实例的特征标签
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1
    baseEntropy=calcshannonEnt(dataSet)#保存最初的原始香农熵作为最初无序度量值
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numFeatures):#遍历所有特征
        #创建唯一的分类标签
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)
        newEntropy=0.0
        #计算每一种划分方式的信息熵
        for value in uniqueVals:#遍历当前特征中的所有唯一属性值
            subDtaSet=splitDataSet(dataSet,i,value)#对每一个唯一属性值划分一次数据集
            prob=1.0*len(subDtaSet)/len(dataSet)#划分得到的数据集的香农熵
            newEntropy+=prob*calcshannonEnt(subDtaSet)#对所有唯一特征值得到的熵值求和
        infoGain=baseEntropy-newEntropy#信息增益为熵的减少或数据无序度的减少
        if (infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

#多数表决决定叶子节点的分类
#当数据集已处理所有属性，但类标签依然不是唯一时调用
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]#返回出现次数最多的分类名称

#创建决策树
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    #类别完全相同则停止划分
    if classList.count(classList[0])==len(classList):
        return classList[0]
    #遍历完所有特征仍不能将数据集划分为包含唯一类别的分组，则返回出现次数最多的类别
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)#当前数据集选取的最好特征
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}#使用字典存储树的信息
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]#得到该特征的所有属性值
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]#复制类标签
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

#获取决策树叶结点的数目
def getNumLeafs(myTree):
    numLeafs=0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#判断节点的数据类型是否为字典
            numLeafs+=getNumLeafs(secondDict[key])
        else:
            numLeafs+=1
    return numLeafs

#获取决策树的层次
def getTreeDepth(myTree):
    maxDepth=0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else:
            thisDepth=1
        if thisDepth>maxDepth:
            maxDepth=thisDepth
    return maxDepth

#在父子节点间填充文本信息
def plotMidText(cntrPt,parentPt,txtString):
    global ax1
    #计算父结点和子节点的中间位置
    xMid=(parentPt[0]-cntrPt[0])/2.0+cntrPt[0]
    yMid=(parentPt[1]-cntrPt[1])/2.0+cntrPt[1]
    ax1.text(xMid,yMid,txtString,color='blue',size=15,rotation=30)#添加文本标签信息

#绘制带箭头的注解
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    global ax1
    ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,
                 textcoords='axes fraction',va="center",ha="center",bbox=nodeType,
                 arrowprops=arrow_args,color='red',size=15)

def plotTree(myTree,parentPt,nodeTxt):
    global ax1,totalW,totalD,x0ff,y0ff
    #计算宽和高
    numLeafs=getNumLeafs(myTree)
    depth=getTreeDepth(myTree)
    firstStr=list(myTree.keys())[0]
    cntrPt=(x0ff+(1.0+float(numLeafs))/2.0/totalW,y0ff)#放在所有叶子结点的中间
    plotMidText(cntrPt,parentPt,nodeTxt)#标记子节点的特征值
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict=myTree[firstStr]
    y0ff=y0ff-1.0/totalD#按比例减少
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:#叶结点
            x0ff=x0ff+1.0/totalW
            plotNode(secondDict[key],(x0ff,y0ff),cntrPt,leafNode)
            plotMidText((x0ff,y0ff),cntrPt,str(key))
    y0ff=y0ff+1.0/totalD

def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')#创建绘图区
    fig.clf()#清空绘图区
    global ax1,totalW,totalD,x0ff,y0ff
    axprops=dict(xticks=[],yticks=[])
    ax1=plt.subplot(111,frameon=False,**axprops)
    totalW=float(getNumLeafs(inTree))#决策树的宽度
    totalD=float(getTreeDepth(inTree))#决策树的深度
    x0ff=-0.5/totalW
    y0ff=1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()

if __name__=='__main__':
    dataArr=loadData('iris.data')
    print(dataArr)
    labels=['sepal length','sepal width','petal length','petal width']
    myTree = createTree(dataArr, labels)
    print(myTree)
    createPlot(myTree)