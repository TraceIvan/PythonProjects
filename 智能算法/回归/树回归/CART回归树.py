import numpy as np
#载入数据，最后一列为y值
def loadDataSet(fileName):
    dataMat=[]
    with open(fileName,'r',encoding='utf8') as fr:
        lines=fr.readlines()
        for line in lines:
            curLine=line.strip().split()
            fltLine=list(map(float,curLine))#将每行映射为浮点数
            dataMat.append(fltLine)
    return np.mat(dataMat)
#按照指定特征及其阈值将数据集划分为2个子集
def binSplitDataSet(dataSet,feature,value):
    '''
    :param dataSet: 数据集合
    :param feature: 待切分的特征
    :param value: 该特征的某个值
    :return: 两个子集
    '''
    mat0=dataSet[np.nonzero(dataSet[:,feature]>value)[0],:]
    mat1=dataSet[np.nonzero(dataSet[:,feature]<=value)[0],:]
    return mat0,mat1
#生成叶结点
def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])
#误差估计函数，计算平方误差总和
def regErr(dataSet):
    return np.var(dataSet[:,-1])*dataSet.shape[0]
#找到最佳的二元切分方式
def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS,tolN=ops[0],ops[1]#自定义容许的误差下降值、切分的最少样本数
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:#剩余样本y取值只有1种情况，无需切分
        return None,leafType(dataSet)
    m,n=dataSet.shape
    S=errType(dataSet)#当前数据集的总平方误差
    bestS,bestIndex,bestValue=np.inf,0,0
    #在所有可能的特征及其可能取值上遍历
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].flatten().A[0]):
            mat0,mat1=binSplitDataSet(dataSet,featIndex,splitVal)
            if mat0.shape[0]<tolN or mat1.shape[0]<tolN:
                continue
            newS=errType(mat0)+errType(mat1)
            if newS<bestS:
                bestS=newS
                bestIndex=featIndex
                bestValue=splitVal

    if (S-bestS)<tolS:#误差下降不明显，则直接创建叶结点
        return None,leafType(dataSet)
    mat0,mat1=binSplitDataSet(dataSet,bestIndex,bestValue)
    if mat0.shape[0] < tolN or mat1.shape[0] < tolN:#如果某个子集的大小小于切分最小样本数，则也不进行切分
        return None,leafType(dataSet)
    return bestIndex,bestValue#返回特征与阈值

#建立树
def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    '''
    :param dataSet: 数据集
    :param leafType: 建立叶结点的函数
    :param errType: 误差计算函数
    :param ops: 树构建所需其他参数的元组
    :return: 树
    '''
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
#判断是否为树
def isTree(obj):
    return (type(obj).__name__=='dict')
#对树进行塌陷处理（返回树平均值）
def getMean(tree):
    if isTree(tree['right']):
        tree['right']=getMean(tree['right'])
    if isTree(tree['left']):
        tree['left']=getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
#对树进行后剪枝
def prune(tree,testData):
    '''
    :param tree: 待剪枝的树
    :param testData: 用于剪枝的测试数据
    :return:
    '''
    if testData.shape[0]==0:#如果测试集为空
        return getMean(tree)
    lSet,rSet=[],[]
    if (isTree(tree['left'])) or (isTree(tree['right'])):
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):#对左子树剪枝
        tree['left']=prune(tree['left'],lSet)
    if isTree(tree['right']):#对右子树剪枝
        tree['right']=prune(tree['right'],rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):#如果左右分支均非树结构，考虑合并
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge=sum(np.power(lSet[:,-1]-tree['left'],2))+sum(np.power(rSet[:,-1]-tree['right'],2))
        treeMean=(tree['left']+tree['right'])/2.0
        errorMerge=sum(np.power(testData[:,-1]-treeMean,2))
        if errorMerge<errorNoMerge:#如果叶结点合并能够降低测试误差，则合并
            print('Merging')
            return treeMean
        else:
            return tree
    else:
        return tree

#回归树叶结点预测
def regTreeEval(model,inDat):
    return float(model)
#对某一条测试数据预测
def treeForeCast(tree,inDat,modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree,inDat)
    if inDat[tree['spInd']]>tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inDat,modelEval)
        else:
            return modelEval(tree['left'],inDat)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inDat,modelEval)
        else:
            return modelEval(tree['right'],inDat)

#回归树预测
def createForeCast(tree,testData,modelEval=regTreeEval):
    m=len(testData)
    yHat=np.mat(np.ones((m,1)))
    for i in range(m):
        yHat[i,0]=treeForeCast(tree,np.mat(testData[i]),modelEval)
    return yHat

#相关系数评价
def corrCoef(yHat,testMat):
    return np.corrcoef(yHat,testMat[:,-1])

if __name__=='__main__':
    myDat=loadDataSet('ex0.txt')
    d1=np.mat(np.eye((4)))
    print((d1[:,-1].T.tolist()))
    myTree=createTree(myDat)
    print(myTree)

    myDat=loadDataSet('ex2.txt')
    myTree=createTree(myDat,ops=(1000,4))
    print(myTree)
    myTest=loadDataSet('ex2test.txt')
    myTree=prune(myTree,myTest)
    print(myTree)

    trainMat=loadDataSet('bikeSpeedVsIq_train.txt')
    testMat=loadDataSet('bikeSpeedVsIq_test.txt')
    myTree=createTree(trainMat,ops=(1,20))
    yHat=createForeCast(myTree,testMat[:,0])
    print(np.corrcoef(yHat,testMat[:,1],rowvar=False)[0,1])
