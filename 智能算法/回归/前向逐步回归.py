import numpy as np
import matplotlib.pyplot as plt
#载入数据
def loadDataSet(fileName):
    numFeat=0
    dataMat,labelMat=[],[]
    with open(fileName,'r',encoding='utf8') as fr:
        lines=fr.readlines()
        for line in lines:
            lineArr=[]
            curLine=line.strip().split()
            numFeat=len(curLine)-1
            for i in range(numFeat):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
    return np.mat(dataMat),np.mat(labelMat).T
#按列标准化
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = np.mean(inMat,0)
    inVar = np.var(inMat,0)
    inMat = (inMat - inMeans)/inVar
    return inMat
#前向逐步回归
def stageWise(xMat,yMat,eps=0.01,numIt=100):
    #标准化
    yMean=np.mean(yMat,0)
    yMat=yMat-yMean
    xMat=regularize(xMat)
    #初始化
    m,n=xMat.shape
    returnMat=np.zeros((numIt,n))
    ws=np.zeros((n,1))
    wsTest,wsMax=ws.copy(),ws.copy()
    for i in range(numIt):
        lowerError=np.inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest=ws.copy()
                wsTest[j]+=eps*sign
                yTest=xMat*wsTest
                rssE=rssError(yMat,yTest)
                if rssE<lowerError:
                    lowerError=rssE
                    wsMax=wsTest
        ws=wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

#计算预测误差
def rssError(yMat,yHat):
    return np.sum(np.power(yMat-yHat,2))
#计算相关系数
def corrCoef(yMat,yHat):
    return np.corrcoef(yMat.T,yHat.T)

if __name__=='__main__':
    xMat, yMat = loadDataSet('abalone.txt')
    weights=stageWise(xMat,yMat,0.001,5000)
    plt.figure()
    plt.plot(weights)
    plt.show()