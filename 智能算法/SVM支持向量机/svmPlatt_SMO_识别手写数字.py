import numpy as np
from os import listdir
import svmPlatt_SMO as svmSMO
#将文本格式的32*32黑白图像转换为1*1024的向量
def imgtxt2vector(filename):
    returnVector=np.zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVector[0,32*i+j]=int(lineStr[j])
    return returnVector

def loadImages(dirName):
    hwLabels=[]
    trainingFileList=listdir(dirName)
    m=len(trainingFileList)
    trainingMat=np.zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        if classNumStr==9:hwLabels.append(-1)#设置9为-1
        else:#其他数字为1
            hwLabels.append(1)
        trainingMat[i,:]=imgtxt2vector('%s/%s'%(dirName,fileNameStr))
    return trainingMat,hwLabels

def testDigits(kTup=('rbf',10)):
    dataArr,labelArr=loadImages('digits/trainingDigits')
    b,alphas=svmSMO.smoP(dataArr,labelArr,200,0.0001,10000,kTup)
    dataMat=np.mat(dataArr)
    labelMat=np.mat(labelArr).transpose()
    svInd=np.nonzero(alphas.A>0)[0]
    sVs=dataMat[svInd]
    labelSV=labelMat[svInd]
    print('there are %d Support Vectors'%(np.shape(sVs)[0]))
    m,n=np.shape(dataMat)
    errorCount=0
    for i in range(m):
        kernelEval=svmSMO.kernelTrans(sVs,dataMat[i,:],kTup)#得到转换后的数据
        predict=kernelEval.T*np.multiply(labelSV,alphas[svInd])+b
        if np.sign(predict)!=np.sign(labelArr[i]):
            errorCount+=1
    print('the training error rate is : %f.'%(float(errorCount)/m))

    dataArr, labelArr =loadImages('digits/testDigits')
    errorCount = 0
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelMat).transpose()
    m, n = np.shape(dataMat)
    for i in range(m):
        kernelEval = svmSMO.kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print('the test error rate is : %f.' % (float(errorCount) / m))

if __name__=='__main__':
    testDigits(('rbf',5))