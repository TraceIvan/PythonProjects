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
    weakClassArr=[]
    m=

