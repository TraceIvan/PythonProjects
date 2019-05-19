import numpy as np

def loadDataSet(fileName):
    dataMat=[]
    with open(fileName) as fr:
        for line in fr.readlines():
            curLine=line.strip().split('\t')
            fltLine=list(map(float,curLine))
            dataMat.append(fltLine)
    return dataMat

def linearSolve(dataSet):
    m,n=np.shape(dataSet)
    X=np.mat(np.ones((m,n)))
    Y=np.mat(np.ones((m,1)))
    X[:,1:n]=dataSet[:,0:n-1]
    Y=dataSet[:,-1]
    xTx=X.T*X
    if np.linalg.det(xTx)==0:
        raise NameError('This matrix is singular, cannot do inverse,\ntry increasing the second value of ops.')
    ws=xTx.I*(X.T*Y)
    return ws,X,Y

def modelLeaf(dataSet):
    ws,X,Y=linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y=linearSolve(dataSet)
    yHat=X*ws
    return np.sum(np.power(Y-yHat,2))

def binSplitDataSet(dataSet,feature,value):
    mat0=dataSet[np.nonzero(dataSet[:,feature]>value)[0],:]
    mat1=dataSet[np.nonzero(dataSet[:,feature]<=value)[0],:]
    return mat0,mat1

def chooseBestSplit(dataSet,leafType=modelLeaf,errType=modelErr,ops=(1,4)):
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


def createTree(dataSet,leafType=modelLeaf,errType=modelErr,ops=(1,4)):
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

if __name__=='__main__':
    myDat=loadDataSet('exp2.txt')
    myMat=np.mat(myDat)
    mytree=createTree(myMat,ops=(1,10))
    print(mytree)