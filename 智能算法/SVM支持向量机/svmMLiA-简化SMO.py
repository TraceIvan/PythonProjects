import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
#打开文件并逐行解析，得到每行的类标签和整个数据矩阵
def loadDataSet(fileName):
    dataMat=[];labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])#前两个为特征
        labelMat.append(float(lineArr[2]))
    fr.close()
    return dataMat,labelMat
#i为第一个alpha的下标，m为所有alpha的数目，随机选择一个下标值
def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j
#调整大于H或小于L的alpha值
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj
#简化SMO跳过外循环寻找最佳alpha对的过程，两次循环随机选择两个alpha构建alpha对
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    '''

    :param dataMatIn:数据集
    :param classLabels: 类别标签
    :param C: 常数C
    :param toler: 容错率
    :param maxIter: 退出前最大循环次数
    :return:b与alphas
    '''
    #转换为numpy矩阵
    dataMatrix=np.mat(dataMatIn)
    labelMat=np.mat(classLabels).transpose()#转换为列向量
    b=0
    m,n=np.shape(dataMatrix)#样例数、特征数
    alphas=np.mat(np.zeros((m,1)))#拉格朗日乘子引入后的约束系数
    iter=0#在没有任何alpha改变的情况下遍历数据集的次数
    while(iter<maxIter):#当在所有数据集遍历naxIter遍后且没有alpha修改才会退出
        alphaPairsChanged=0#记录alpha是否已经优化
        for i in range(m):
            fXi=float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b#预测的i的类别
            Ei=fXi-float(labelMat[i])#与实际值的误差
            if((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or ((labelMat[i]*Ei>toler) and (alphas[i]>0)):#误差较大则进行优化
                #如果alpha为0或C，已经位于边界上，无需优化
                j=selectJrand(i,m)#随机得到第二个alpha
                fXj=float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b#预测的j的类型
                Ej=fXj-float(labelMat[j])#j的实际值的误差
                alphaIOld=alphas[i].copy()#记录以前的i,j的alpha
                alphaJOld=alphas[j].copy()
                #计算L和H，用于调整alpha[j]到0~C之间
                if (labelMat[i]!=labelMat[j]):
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                if L==H:#L和H相同，则不做改变
                    print("L==H")
                    continue
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T#alphas[j]的最优修改量
                if eta>=0:#eta为0，退出本次循环(Platt的SMO算法中有对eta为0重新计算alpha[j]的步骤，此处忽略)
                    print("eta>=0")
                    continue
                #计算新的alpha[j]
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                alphas[j]=clipAlpha(alphas[j],H,L)
                if (abs(alphas[j]-alphaJOld)<0.00001):#只有轻微的改变
                    print("j not moving enough")
                    continue
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJOld-alphas[j])#alpha[i]同样改变，但改变方向和alpha[j]相反(一个增加，一个减少)
                #重新设置常数b
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIOld)*dataMatrix[i,:]*dataMatrix[i,:].T\
                   -labelMat[j]*(alphas[j]-alphaJOld)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIOld)*dataMatrix[i,:]*dataMatrix[j,:].T\
                   -labelMat[j]*(alphas[j]-alphaJOld)*dataMatrix[j,:]*dataMatrix[j,:].T
                if(0<alphas[i]) and (C>alphas[i]):
                    b=b1
                elif (0<alphas[j]) and (C>alphas[j]):
                    b=b2
                else:
                    b=(b1+b2)/2
                alphaPairsChanged+=1
                print('iter: %d i: %d,pairs changed %d'%(iter,i,alphaPairsChanged))
        if (alphaPairsChanged==0):
            iter+=1
        else:#如果有更新alpha，置为0
            iter=0
        print("iteration number: %d"%iter)
    return b,alphas
#得到超平面系数w
def calcWs(alphas,dataArr,classLabels):
    X=np.mat(dataArr)
    labelMat=np.mat(classLabels).transpose()
    m,n=np.shape(X)
    w=np.zeros((n,1))
    for i in range(m):
        w+=np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

#对数据集分类
def classifySVM(dataArr,Ws,b):
    dataMat=np.mat(dataArr)
    n=np.shape(dataArr)[0]
    clf_re=np.zeros((n,1))
    for i in range(n):
        value=dataMat[i]*Ws+b
        if value<0:
            clf_re[i]=-1
        elif value>0:
            clf_re[i]=1
        else:
            clf_re[i]=0
    return clf_re

def plotSupportVectors(Ws,b,alphas,dataArr,labelArr):
    dataMatrix=np.mat(dataArr)
    labelMat=np.mat(labelArr)
    (m,n)=np.shape(dataArr)
    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []
    markers = []
    colors = []
    for i in range(m):
        if labelMat[0,i]==-1:
            xcord0.append(dataMatrix[i,0])
            ycord0.append(dataMatrix[i,1])
        else:
            xcord1.append(dataMatrix[i,0])
            ycord1.append(dataMatrix[i,1])


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord0, ycord0, marker='s', s=90)
    ax.scatter(xcord1, ycord1, marker='o', s=50, c='red')
    plt.title('Support Vectors Circled')
    spVec=np.nonzero(alphas)[0]
    for i in spVec:
        xi=dataMatrix[i,0]
        yi=dataMatrix[i,1]
        circle = Circle((xi,yi), 0.5, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3,
                    alpha=0.5)
        ax.add_patch(circle)
    # plot seperating hyperplane
    w0 = Ws[0,0]
    w1 = Ws[1,0]
    x = np.arange(-2.0, 12.0, 0.1)
    y = np.array((-w0 * x - b) / w1)[0]
    ax.plot(x, y)
    plt.axis([-2, 12, -8, 6])
    plt.show()
if __name__=='__main__':
    dataArr,labelArr=loadDataSet('testSet.txt')
    b,alphas=smoSimple(dataArr,labelArr,0.6,0.001,40)
    print(b)
    print(alphas[alphas>0])
    print(np.shape(alphas))
    (tn,_)=np.shape(alphas)
    for i in range(tn):
        if alphas[i]>0.0:
            print(dataArr[i],labelArr[i])
    Ws=calcWs(alphas,dataArr,labelArr)
    print(Ws)
    re=classifySVM(dataArr,Ws,b)
    n=np.shape(np.mat(dataArr))[0]
    for i in range(n):
        print('classified answer is %f, true answer is %f.'%(re[i],labelArr[i]))
    plotSupportVectors(Ws,b,alphas,dataArr,labelArr)

