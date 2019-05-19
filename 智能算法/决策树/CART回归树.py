import numpy as np

def loadDataSet(fileName):
    dataMat=[]
    with open(fileName) as fr:
        for line in fr.readlines():
            curLine=line.strip().split('\t')
            fltLine=list(map(float,curLine))
            dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet,feature,value):
    mat0=dataSet[np.nonzero(dataSet[:,feature]>value)[0],:]
    mat1=dataSet[np.nonzero(dataSet[:,feature]<=value)[0],:]
    return mat0,mat1

def regLeaf(dataSet):#生成叶结点
    return np.mean(dataSet[:,-1])

def regErr(dataSet):#误差估计函数
    return np.var(dataSet[:,-1])*np.shape(dataSet)[0]#平方误差(总方差=均方差*样本个数)

def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    '''找到最佳二元切分方式'''
    tolS=ops[0]#容许的误差下降值
    tolN=ops[1]#切分的最少样本数
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:#所有值相等
        return None,leafType(dataSet)
    m,n=np.shape(dataSet)
    S=errType(dataSet)
    bestS,bestIndex,bestValue=np.inf,0,0
    for featIndex in range(n-1):
        for splitVal in set(np.reshape(dataSet[:,featIndex],(1,-1)).tolist()[0]):
            mat0,mat1=binSplitDataSet(dataSet,featIndex,splitVal)
            if (np.shape(mat0)[0]<tolN) or (np.shape(mat1)[0]<tolN):
                continue
            newS=errType(mat0)+errType(mat1)
            if newS<bestS:
                bestIndex=featIndex
                bestValue=splitVal
                bestS=newS
    if (S-bestS)<tolS:#如果误差减少不大
        return None,leafType(dataSet)
    mat0,mat1=binSplitDataSet(dataSet,bestIndex,bestValue)
    if (np.shape(mat0)[0]<tolN) or (np.shape(mat1)[0]<tolN):#如果切分出的数据集很小
        return None,leafType(dataSet)
    return bestIndex,bestValue


def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,val=chooseBestSplit(dataSet,leafType,errType,ops)
    if feat==None:
        return val
    retTree={}
    retTree['spInd']=feat
    retTree['spVal']=val
    lSet,rSet=binSplitDataSet(dataSet,feat,val)
    retTree['left']=createTree(lSet,leafType,errType,ops)
    retTree['right']=createTree(rSet,leafType,errType,ops)
    return retTree
#回归树后剪枝
def isTree(obj):
    return type(obj).__name__=='dict'

def getMean(tree):
    if isTree(tree['right']):
        tree['right']=getMean(tree['right'])
    if isTree(tree['left']):
        tree['left']=getMean(tree['left'])
    return (tree['left']+tree['right'])/2

def prune(tree,testData):
    if np.shape(testData)[0]==0:
        return getMean(tree)
    lSet,rSet=[],[]
    if isTree(tree['right']) or isTree(tree['left']):
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):
        tree['left']=prune(tree['left'],lSet)
    if isTree(tree['right']):
        tree['right']=prune(tree['right'],rSet)

    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge=np.sum(np.power(lSet[:,-1]-tree['left'],2))+np.sum(np.power(rSet[:,-1]-tree['right'],2))
        treeMean=(tree['left']+tree['right'])/2
        errorMerge=np.sum(np.power(testData[:,-1]-treeMean,2))
        if errorMerge < errorNoMerge:#比较合并前后的误差
            print("Merging")
            return treeMean
        else:
            return tree
    else:
        return tree


if __name__=='__main__':
    myDat=loadDataSet('ex0.txt')#ex0.txt,ex2.txt
    #print(myDat)
    myMat=np.mat(myDat)
    print(myMat)
    mytree=createTree(myMat)
    print(mytree)
    myDat2=loadDataSet('ex2.txt')
    myMat2=np.mat(myDat2)
    tree2=createTree(myMat2,ops=(0,1))
    myDat2_test=loadDataSet('ex2test.txt')
    myMat2_test=np.mat(myDat2_test)
    tree2_prune=prune(tree2,myMat2_test)
    print(tree2_prune)