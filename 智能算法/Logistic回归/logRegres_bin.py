#二分类logistic回归
import numpy as np
import random
import matplotlib.pyplot as plt
#读取测试文件
def loadDataSet():
    dataMat=[]
    labelMat=[]
    with open('testSet.txt','rb') as fr:
        for line in fr.readlines():
            linArr=line.strip().split()
            dataMat.append([1.0,float(linArr[0]),float(linArr[1])])#每行前两个为X1,X2特征值，X0置为1
            labelMat.append(int(linArr[2]))#第三个为类别标签
    return np.mat(dataMat),np.mat(labelMat)
#sigmoid函数
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))
#梯度上升法
def gradAscent(dataMatIn,classLabels):
    dataMatrix=np.mat(dataMatIn)#转换为numpy矩阵
    labelMat=np.mat(classLabels).transpose()#转换为列向量
    m,n=np.shape(dataMatrix)
    alpha=0.001#步长
    maxCycles=100000#迭代次数
    weights=np.ones([n,1])
    x0 = [1];x1 = [1];x2 = [1]
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=(labelMat-h)#计算真实类别与预测类别的差值
        weights=weights+alpha*dataMatrix.transpose()*error
        x0.append(weights[0, 0]);x1.append(weights[1, 0]);x2.append(weights[2, 0])
    return weights,x0,x1,x2
#随机梯度上升
def stocGradAscent0(dataMatrix,classLabels,numIter=200):
    '''输入均为mat'''
    m,n=np.shape(dataMatrix)
    weights=np.ones([n,1])
    x0=[1];x1=[1];x2=[1]
    for j in range(numIter):
        for i in range(m):
            alpha = 0.01+4/(1.0+i+j)#alpha每次迭代时进行调整
            randIndex=int(random.uniform(0,m))#随机选择一个样例进行计算来更新回归系数
            g=sigmoid(dataMatrix[randIndex]*weights)
            error=classLabels[0,randIndex]-g
            weights=weights+dataMatrix[randIndex].transpose()*alpha*error
            x0.append(weights[0,0]);x1.append(weights[1,0]);x2.append(weights[2,0])
    return weights,x0,x1,x2
#画出最佳拟合直线
def show(dataMat,labelMat,weights):
    m,n=np.shape(dataMat)
    min_x=np.amin(dataMat,axis=0)[0,1]
    max_x=np.max(dataMat,axis=0)[0,1]
    xcoord1=[];ycoord1=[]
    xcoord2=[];ycoord2=[]
    for i in range(m):
        if int(labelMat[0,i])==0:
            xcoord1.append(dataMat[i,1])
            ycoord1.append(dataMat[i,2])
        else:
            xcoord2.append(dataMat[i,1])
            ycoord2.append(dataMat[i,2])
    plt.figure()
    plt.scatter(xcoord1,ycoord1,s=15,c='red')
    plt.scatter(xcoord2,ycoord2,s=15,c='blue')
    x=np.arange(min_x,max_x,0.1)
    y=(-float(weights[0])-float(weights[1])*x)/float(weights[2])
    plt.plot(x,y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def show2(x0,x1,x2):
    plt.figure(2)
    counts=len(x0)
    counts=np.arange(0,counts,1)
    plt.subplot(311)
    plt.plot(counts,x0)
    plt.ylabel('x0')
    plt.subplot(312)
    plt.plot(counts,x1)
    plt.ylabel('x1')
    plt.subplot(313)
    plt.plot(counts,x2)
    plt.ylabel('x2')
    plt.show()
if __name__=='__main__':
    dataMat,labelMat=loadDataSet()
    weights,x0,x1,x2=gradAscent(dataMat,labelMat)
    print(labelMat[0,1])
    show(dataMat,labelMat,weights)
    show2(x0, x1, x2)
    weights2,x0,x1,x2=stocGradAscent0(dataMat,labelMat)
    show(dataMat,labelMat,weights2)
    show2(x0,x1,x2)