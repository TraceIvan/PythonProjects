import numpy as np

def loadSimpData():
    dataMat=np.matrix([[1.,2.1],
                       [2.,1.1],
                       [1.,1.],
                       [1.3,1.],
                       [2.,1.]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels

#单层决策树（decision stump）生成函数
#通过阈值比较对数据分类
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray=np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq=='lt':
        retArray[dataMatrix[:,dimen]<=threshVal]=-1.0#通过数组过滤实现
    else:
        retArray[dataMatrix[:,dimen]>threshVal]=-1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    '''
    :param dataArr:
    :param classLabels:
    :param D:
    :return:返回利用D得到的具有最小错误率的单层决策树、最小的错误率、估计的类别向量
    '''
    dataMatrix=np.mat(dataArr)
    labelMat=np.mat(classLabels).T
    m,n=np.shape(dataMatrix)
    numSteps=10.0#在特征的所有可能值上遍历
    bestStump={}#存放给定权重向量D时得到的最佳单层决策树
    bestClasEst=np.mat(np.zeros((m,1)))
    minError=np.inf#最小错误率
    for i in range(n):#找到最佳单层决策树，在所有特征上进行遍历
        rangeMin=dataMatrix[:,i].min()
        rangeMax=dataMatrix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps#计算步长
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:#在小于和大于之间切换
                threshVal=(rangeMin+float(j)*stepSize)
                predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr=np.mat(np.ones((m,1)))#表示预测是否和真实标签一样
                errArr[predictedVals==labelMat]=0#一样的置为0
                weightedError=D.T*errArr#计算加权错误率
                print('split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f'%(i,threshVal,inequal,weightedError))

                if weightedError<minError:
                    minError=weightedError
                    bestClasEst=predictedVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    return bestStump,minError,bestClasEst

#基于单层决策树的AdaBoost训练过程
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    '''
    :param dataArr:数据集
    :param classLabels: 类别标签
    :param numIt: 迭代次数
    :return: 单层决策树的数组
    '''
    weakClassArr=[]
    m=np.shape(dataArr)[0]
    D=np.mat(np.ones((m,1))/m)#包含每个数据点的权重，初始值相同
    aggClassEst=np.mat(np.zeros((m,1)))#每个数据点的类别估计累计值
    for i in range(numIt):#运行numIt次或直到错误率为0
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)#建立单层决策树
        print("D: ",D.T)
        alpha=float(0.5*np.log((1.0-error)/max(error,1e-16)))#计算alpha,即本次单层决策树输出结果的权重，max(error,1e-16)确保没有错误时不会发生除0溢出
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump)
        print("classEst: ",classEst.T)
        #为下一次迭代计算D
        expon=np.multiply(-1*alpha*np.mat(classLabels).T,classEst)
        D=np.multiply(D,np.exp(expon))
        D=D/D.sum()
        #错误率累加计算
        aggClassEst+=alpha*classEst#保持运行时的类别估计值
        print("aggClassEst ：",aggClassEst.T)
        aggErrors=np.multiply(np.sign(aggClassEst)!=np.mat(classLabels).T,np.ones((m,1)))
        errorRate=aggErrors.sum()/m
        print("total error: ",errorRate)
        if errorRate==0.0:
            break
    return weakClassArr
#AdaBoost分类函数,即利用训练出的多个弱分类器进行分类
def adaClassify(datToClass,classifierArr):
    dataMatrix=np.mat(datToClass)
    m=np.shape(dataMatrix)[0]
    aggClassEst=np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):#遍历所有弱分类器
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])#得到类别估计值
        aggClassEst+=classifierArr[i]['alpha']*classEst#类别估计值*该单层决策树的alpha权重
        print(aggClassEst)
    return np.sign(aggClassEst)#返回符号
if __name__=='__main__':
    datMat,classLabels=loadSimpData()
    classifierArray=adaBoostTrainDS(datMat,classLabels,9)
    print(classifierArray)
    classifiedResult=adaClassify(datMat,classifierArray)
    print(classifiedResult)
    print(classLabels)


