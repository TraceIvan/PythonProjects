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
#岭回归
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx=np.dot(xMat.T,xMat)
    denom=xTx+np.mat(np.eye(xMat.shape[1])*lam)
    if np.linalg.det(denom)==0.0:
        print("This matrix is singular, can't do inverse.")
        return -1
    ws=np.dot(denom.I,np.dot(xMat.T,yMat))
    return ws
#寻找最佳λ系数
def ridgeTest(xMat,yMat):
    #数据标准化
    yMean=np.mean(yMat,0)
    yMat=yMat-yMean
    xMeans=np.mean(xMat,0)
    xVar=np.var(xMat,0)
    xMat=(xMat-xMeans)/xVar
    #测试岭回归
    numTestPts=30
    E=[]
    lamArr=[]
    wMat=np.zeros((numTestPts,xMat.shape[1]))
    for i in range(numTestPts):
        ws=ridgeRegres(xMat,yMat,np.exp(i-10))
        wMat[i:,]=ws.T
        yHat=xMat*ws
        E.append(rssError(yMat,yHat))
        lamArr.append(i-10)
    return np.mat(wMat),E,lamArr


#计算预测误差
def rssError(yMat,yHat):
    return np.sum(np.power(yMat-yHat,2))

#计算相关系数
def corrCoef(yMat,yHat):
    return np.corrcoef(yMat.T,yHat.T)

#绘制岭迹图
def show(wMat,lamArr,colorList):
    m,n=wMat.shape
    plt.figure()
    for i in range(n):
        plt.plot(lamArr,wMat[:,i],colorList[i])
    plt.show()


if __name__=='__main__':
    xMat, yMat = loadDataSet('abalone.txt')
    wMat,E,lamArr=ridgeTest(xMat,yMat)
    colorList=['#DC143C','#EE8262','#CD9B1D','#CD00CD','#458B00','#3B3B3B','#00FF00','#0000FF']
    show(wMat,lamArr,colorList)
    plt.figure(1)
    plt.plot(lamArr,E)
    plt.show()
