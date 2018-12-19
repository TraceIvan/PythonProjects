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
#对数据集进行OLS线性回归
def linearSolve(dataMat):
    m,n=dataMat.shape
    X,Y=np.mat(np.ones((m,n))),np.mat(np.ones((m,1)))
    X[:,1:n]=dataMat[:,0:n-1]#第一列为常数项
    Y=dataMat[:,-1]
    xTx=np.dot(X.T,X)
    if np.linalg.det(xTx)==0.0:
        raise NameError("This matrix is singular, can't do inverse.\nTry increasing the second value of ops.")
    ws=np.dot(xTx.I,np.dot(X.T,Y))
    return ws,X,Y
#模型树建立叶结点
def modelLeaf(dataMat):
    ws,X,Y=linearSolve(dataMat)
    return ws
#模型树误差计算
def modelErr(dataMat):
    ws,X,Y=linearSolve(dataMat)
    yHat=np.dot(X,ws)
    return np.sum(np.power(Y-yHat,2))
#找到最佳的二元切分方式
def chooseBestSplit(dataSet,leafType=modelLeaf,errType=modelErr,ops=(1,4)):
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
def createTree(dataSet,leafType=modelLeaf,errType=modelErr,ops=(1,4)):
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
# 模型树叶结点预测
def modelTreeEval(model, inDat):
    n=inDat.shape[1]
    X=np.mat(np.ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(np.dot(X,model))


# 对某一条测试数据预测
def treeForeCast(tree, inDat, modelEval=modelTreeEval):
    if not isTree(tree):
        return modelEval(tree, inDat)
    if inDat[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inDat, modelEval)
        else:
            return modelEval(tree['left'], inDat)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inDat, modelEval)
        else:
            return modelEval(tree['right'], inDat)


# 模型树预测
def createForeCast(tree, testData, modelEval=modelTreeEval):
    m=len(testData)
    yHat=np.mat(np.ones((m,1)))
    for i in range(m):
        yHat[i,0]=treeForeCast(tree,np.mat(testData[i]),modelEval)
    return yHat

#相关系数评价
def corrCoef(yHat,testMat):
    return np.corrcoef(yHat,testMat[:,-1])

if __name__=='__main__':
    data1=loadDataSet('exp2.txt')
    tree1=createTree(data1,modelLeaf,modelErr,(1,10))
    print(tree1)

    trainMat=loadDataSet('bikeSpeedVsIq_train.txt')
    testMat=loadDataSet('bikeSpeedVsIq_test.txt')
    myTree=createTree(trainMat,ops=(1,20))
    yHat=createForeCast(myTree,testMat[:,0])
    print(np.corrcoef(yHat,testMat[:,1],rowvar=False)[0,1])