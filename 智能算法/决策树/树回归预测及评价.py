import CART回归树 as cart_reg
import CART模型树 as cart_model
import numpy as np
#预测
def isTree(obj):
    return type(obj).__name__=='dict'

def regTreeEval(model,inDat):
    return float(model)

def modelTreeEval(model,inDat):
    n=np.shape(inDat)[1]
    X=np.mat(np.ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree,inData,modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree,inData)
    if inData[tree['spInd']]>tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)

def createForeCast(tree,testData,modelEval=regTreeEval):
    m=np.shape(testData)[0]
    yHat=np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat[i,0]=treeForeCast(tree,np.mat(testData[i]),modelEval)
    return yHat

def linear_ForeCast(testData):
    m = np.shape(testData)[0]
    yHat = np.mat(np.zeros((m, 1)))
    ws, X, Y = cart_model.linearSolve(trainMat)
    for i in range(m):
        yHat[i]=testData[i,0]*ws[1,0]+ws[0,0]
    return yHat
if __name__=='__main__':
    trainMat=np.mat(cart_reg.loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = np.mat(cart_reg.loadDataSet('bikeSpeedVsIq_test.txt'))
    mytree=cart_reg.createTree(trainMat,ops=(1,20))
    yHat=createForeCast(mytree,testMat[:,0])#回归树
    corr1=np.corrcoef(yHat,testMat[:,1],rowvar=0)#R2相关系数，越接近1越好
    print(corr1[0,1])
    mytree2 = cart_model.createTree(trainMat, ops=(1, 20))
    yHat2 = createForeCast(mytree2, testMat[:, 0],modelTreeEval)  # 模型树
    corr2 = np.corrcoef(yHat2, testMat[:, 1], rowvar=0)
    print(corr2[0, 1])
    yHat3 = linear_ForeCast(testMat[:, 0])#线性回归
    corr3 = np.corrcoef(yHat3, testMat[:, 1], rowvar=0)
    print(corr3[0, 1])
