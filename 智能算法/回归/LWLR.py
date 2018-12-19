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


#LWLR，对于给定特征空间中的一个数据，计算预测值
def lwlr(testMat,xMat,yMat,k=1.0):
    '''
    :param testMat:行向量，表示给定的特征数据
    :param xMat: 样本
    :param yMat: 样本值
    :param k: 高斯核系数
    :return:
    '''
    m=xMat.shape[0]
    dWeights=np.mat(np.eye(m))
    for j in range(m):
        diffMat=testMat-xMat[j,:]
        dWeights[j,j]=np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx=np.dot(xMat.T,np.dot(dWeights,xMat))
    if np.linalg.det(xTx)==0.0:
        print("This matrix is singular, can't do inverse.")
        return -1
    ws=np.dot(xTx.I,np.dot(xMat.T,np.dot(dWeights,yMat)))
    return np.dot(testMat,ws)

def lwlrSolve(testMat,xMat,yMat,k=1.0):
    m=xMat.shape[0]
    yHat=np.mat(np.zeros(m)).T
    for i in range(m):
        yHat[i,0]=lwlr(testMat[i,:],xMat,yMat,k)
    return yHat

#展示数据和线性回归结果
def show(xMat,yMat,yHat):
    if xMat.shape[1] != 2:
        return
    fig=plt.figure(1)
    srtInd=xMat[:,1].argsort(0)
    xSort=xMat[srtInd][:,0,:]
    #print(yHat[srtInd][:,0,:])
    plt.plot(xSort[:,1],yHat[srtInd][:,0,:])
    plt.scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0],s=2,c='red')
    plt.show()
    corr=np.corrcoef(yHat.T,yMat.T)#改为行向量，计算相关系数
    print("相关系数为:\n",corr)

#计算预测误差
def rssError(yMat,yHat):
    return np.sum(np.power(yMat-yHat,2))
#计算相关系数
def corrCoef(yMat,yHat):
    return np.corrcoef(yMat.T,yHat.T)

if __name__=='__main__':
    xMat,yMat=loadDataSet('ex1.txt')
    yHat=lwlrSolve(xMat,xMat,yMat,0.003)
    show(xMat,yMat,yHat)
    xMat,yMat=loadDataSet('abalone.txt')
    m=xMat.shape[0]
    yHat=lwlrSolve(xMat[0:100],xMat[0:100],yMat[0:100],0.1)
    print(rssError(yMat[0:100],yHat))
    yHat=lwlrSolve(xMat[0:100],xMat[0:100],yMat[0:100],1)
    print(rssError(yMat[0:100],yHat))
    yHat=lwlrSolve(xMat[0:100],xMat[0:100],yMat[0:100],10)
    print(rssError(yMat[0:100],yHat))
    yHat = lwlrSolve(xMat[100:200], xMat[0:100], yMat[0:100], 0.1)
    print(rssError(yMat[100:200], yHat))
    yHat = lwlrSolve(xMat[100:200], xMat[0:100], yMat[0:100], 1)
    print(rssError(yMat[100:200], yHat))
    yHat = lwlrSolve(xMat[100:200], xMat[0:100], yMat[0:100], 10)
    print(rssError(yMat[100:200], yHat))

