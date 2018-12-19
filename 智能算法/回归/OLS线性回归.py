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


#使用OLS方法进行线性回归
def standRegres(xMat,yMat):
    xTx=np.dot(xMat.T,xMat)
    if np.linalg.det(xTx)==0.0:
        print("This matrix is singular, can't do inverse.")
        return -1
    ws=np.dot(xTx.I,np.dot(xMat.T,yMat))
    return ws

#展示数据和线性回归结果
def show(xMat,yMat,ws):
    fig=plt.figure(1)
    plt.scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0])
    xCopy=xMat.copy()
    xCopy.sort(0)#按列排序
    yHat=np.dot(xCopy,ws)
    plt.plot(xCopy[:,1],yHat)
    plt.show()
    yHat=np.dot(xMat,ws)
    corr=np.corrcoef(yHat.T,yMat.T)#改为行向量，计算相关系数
    print("相关系数为:\n",corr)
#计算预测误差
def rssError(yMat,yHat):
    return np.sum(np.power(yMat-yHat,2))
#计算相关系数
def corrCoef(yMat,yHat):
    return np.corrcoef(yMat.T,yHat.T)
if __name__=='__main__':
    mat1=np.mat([[1],[4]])
    print(mat1.flatten().A)
    xMat,yMat=loadDataSet('ex1.txt')
    ws=standRegres(xMat,yMat)
    if ws.shape[0]==2:
        show(xMat,yMat,ws)


