import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

#建立一个数据结构保存所有值
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):#kTup为包含核函数信息的元组
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=np.shape(dataMatIn)[0]
        self.alphas=np.mat(np.zeros((self.m,1)))
        self.b=0
        self.eCache=np.mat(np.zeros((self.m,2)))#缓存，保存误差值。第一列表示是否有效的标志位，第二类为实际的Ei值
#对于给定的k,计算与实际值的误差
        self.K=np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i]=kernelTrans(self.X,self.X[i,:],kTup)

def kernelTrans(X,A,kTup):
    '''
    :param X:
    :param A:
    :param kTup: 描述核函数的信息
    第一个参数描述核函数类型，另外两个参数为可能的可选参数
    :return:
    '''
    m,n=np.shape(X)
    K=np.mat(np.zeros((m,1)))
    if kTup[0]=='lin':
        K=X*A.T
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow=X[j,:]-A
            K[j]=deltaRow*deltaRow.T
        K=np.exp(K/(-1*kTup[1]**2))#使用径向基函数这一核函数映射。此处为元素间的除法
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized.')
    return K
def calcEk(oS,k):
    #fXk=float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T))+oS.b#未使用核函数
    fXk=float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[:,k]+oS.b)#使用核函数
    Ek=fXk-float(oS.labelMat[k])
    return Ek
#i为第一个alpha的下标，m为所有alpha的数目，随机选择一个下标值
def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j
#选择第二个(内循环)合适的alpha的值，保证每次优化中采用最大步长
def selectJ(i,oS,Ei):
    maxK=-1
    maxDeltaE=0
    Ej=0
    oS.eCache[i]=[1,Ei]#首先将输入的Ei设置为有效(表示已经计算完毕)
    validEcacheList=np.nonzero(oS.eCache[:,0].A)[0]#构建一个非零表,返回非零E值对应的alpha值(下标)
    if (len(validEcacheList))>1:#选择具有最大步长的j
        for k in validEcacheList:
            if k==i:
                continue
            Ek=calcEk(oS,k)
            deltaE=abs(Ei-Ek)
            if (deltaE>maxDeltaE):
                maxK=k
                maxDeltaE=deltaE
                Ej=Ek
        return maxK,Ej
    else:#如果为第一次循环，随机选择一个
        j=selectJrand(i,oS.m)
        Ej=calcEk(oS,j)
    return j,Ej
#计算误差并且存入缓存
def updateEk(oS,k):
    Ek=calcEk(oS,k)
    oS.eCache[k]=[1,Ek]
#调整大于H或小于L的alpha值
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj
#寻找决策边界
def innerL(i,oS):
    Ei=calcEk(oS,i)
    if((oS.labelMat[i]*Ei<-oS.tol) and (oS.alphas[i]<oS.C)) or ((oS.labelMat[i]*Ei>oS.tol) and (oS.alphas[i]>0)):#误差较大则进行优化
        #如果alpha为0或C，已经位于边界上，无需优化
        j,Ej=selectJ(i,oS,Ei)#j的实际值的误差
        alphaIOld=oS.alphas[i].copy()#记录以前的i,j的alpha
        alphaJOld=oS.alphas[j].copy()
        # 计算L和H，用于调整alpha[j]到0~C之间
        if(oS.labelMat[i]!=oS.labelMat[j]):
            L=max(0,oS.alphas[j]-oS.alphas[i])
            H=min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L=max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
            H=min(oS.C,oS.alphas[j]+oS.alphas[i])
        if L==H:#L和H相同，则不做改变
            print("L==H")
            return 0
        #eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T#alphas[j]的最优修改量，未使用核函数
        eta=2.0*oS.K[i,j]-oS.K[i,i]-oS.K[j,j]#使用核函数
        if eta>=0:#eta为0，退出本次循环(Platt的SMO算法中有对eta为0重新计算alpha[j]的步骤，此处忽略)
            print("eta>=0")
            return 0
        # 计算新的alpha[j]
        oS.alphas[j]-=oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)#更新误差缓存
        if (abs(oS.alphas[j]-alphaJOld)<0.00001):#只有轻微的改变
            print('j not moving enough')
            return 0
        oS.alphas[i]+=oS.labelMat[j]*oS.labelMat[i]*(alphaJOld-oS.alphas[j])#alpha[i]同样改变，但改变方向和alpha[j]相反(一个增加，一个减少)
        updateEk(oS,i)
        # 重新设置常数b
        #未使用核函数
        #b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIOld) * oS.X[i, :] * oS.X[i, :].T \
        #     - oS.labelMat[j] * (oS.alphas[j] - alphaJOld) * oS.X[i, :] * oS.X[j, :].T
        #b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIOld) * oS.X[i, :] * oS.X[j, :].T \
        #     - oS.labelMat[j] * (oS.alphas[j] - alphaJOld) * oS.X[j, :] * oS.X[j, :].T
        #使用核函数
        b1=oS.b-Ei-oS.labelMat[i] * (oS.alphas[i] - alphaIOld)*oS.K[i,i]-oS.labelMat[j] * (oS.alphas[j] - alphaJOld) *oS.K[i,j]
        b2=oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIOld) *oS.K[i,j] - oS.labelMat[j] * (oS.alphas[j] - alphaJOld) *oS.K[j,j]

        if (0<oS.alphas[i]) and (oS.C>oS.alphas[i]):
            oS.b=b1
        elif (0<oS.alphas[j]) and (oS.C>oS.alphas[j]):
            oS.b=b2
        else:
            oS.b=(b1+b2)/2
        return 1
    else:
        return 0
#外循环
def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    '''
    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :param C: 常数C
    C一方面保障所有样例的间隔不小于1.0，另一方面使得分类间隔尽可能大，并且在这两者之间平衡
    如果C很大，分类器会力图通过分类器使得所有样例都分类正确
    :param toler: 容错率
    :param maxIter: 退出前最大循环次数
    :param kTup:
    :return: b与alphas
    '''
    oS=optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler,kTup)
    iter=0
    entireSet=True
    alphaPairsChanged=0
    while(iter<maxIter) and ((alphaPairsChanged>0) or (entireSet)):
        alphaPairsChanged=0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged+=innerL(i,oS)
                print('fullSet, iter :%d i: %d, pairs changed %d'%(iter,i,alphaPairsChanged))
            iter+=1
        else:
            nonBoundIs=np.nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged+=innerL(i,oS)
                print('non-bound, iter: %d i: %d, pairs changed %d'%(iter,i,alphaPairsChanged))
            iter+=1
        if entireSet:
            entireSet=False
        elif (alphaPairsChanged==0):
            entireSet=True
        print("iteration number: %d"%iter)
    return oS.b,oS.alphas

#得到超平面系数w
def calcWs(alphas,dataArr,classLabels):
    X=np.mat(dataArr)
    labelMat=np.mat(classLabels).transpose()
    m,n=np.shape(X)
    w=np.zeros((n,1))
    for i in range(m):
        w+=np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w
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
    ax.scatter(xcord0, ycord0, marker='s')
    ax.scatter(xcord1, ycord1, marker='o', c='red')
    plt.title('Support Vectors Circled')
    spVec=np.nonzero(alphas)[0]
    for i in spVec:
        xi=dataMatrix[i,0]
        yi=dataMatrix[i,1]
        circle = Circle((xi,yi), 0.3, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3,
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
    b,alphas=smoP(dataArr,labelArr,100,0.0001,400)
    print(b)
    print(alphas)
    Ws=calcWs(alphas,dataArr,labelArr)
    print(Ws)
    re=classifySVM(dataArr,Ws,b)
    n=np.shape(np.mat(dataArr))[0]
    for i in range(n):
        print('classified answer is %f, true answer is %f.'%(re[i],labelArr[i]))
    plotSupportVectors(Ws,b,alphas,dataArr,labelArr)
