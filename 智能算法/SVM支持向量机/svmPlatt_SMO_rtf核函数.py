import svmPlatt_SMO as svmSMO
import numpy as np
def testRbf(k1=1.3):
    '''
    :param k1:径向基函数的自定义的确定到达率(reach)或者函数值跌落到0的速度参数
    :return:
    '''
    dataArr,labelArr=svmSMO.loadDataSet('testSetRBF.txt')
    b,alphas,Ks=svmSMO.smoP(dataArr,labelArr,200,0.0001,100,('rbf',k1))
    print(Ks)
    dataMat=np.mat(dataArr)
    labelMat=np.mat(labelArr).transpose()
    svInd=np.nonzero(alphas.A>0)[0]
    sVs=dataMat[svInd]#构建支持向量矩阵
    labelSV=labelMat[svInd]#对应的类别标签
    print('there are %d Support Vectors.'%np.shape(sVs)[0])
    m,n=np.shape(dataMat)
    errorCount=0
    #利用核函数进行分类(只需要支持向量数据进行分类)
    for i in range(m):
        kernelEval=svmSMO.kernelTrans(sVs,dataMat[i,:],('rbf',k1))#得到转换后的数据
        predict=kernelEval.T*np.multiply(labelSV,alphas[svInd])+b
        if np.sign(predict)!=np.sign(labelArr[i]):
            errorCount+=1
    print('the training error rate is : %f.'%(float(errorCount)/m))

    dataArr,labelArr=svmSMO.loadDataSet('testSetRBF2.txt')
    errorCount=0
    dataMat=np.mat(dataArr)
    labelMat=np.mat(labelMat).transpose()
    m,n=np.shape(dataMat)
    for i in range(m):
        kernelEval=svmSMO.kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict=kernelEval.T*np.multiply(labelSV,alphas[svInd])+b
        if np.sign(predict)!=np.sign(labelArr[i]):
            errorCount+=1
    print('the test error rate is : %f.'%(float(errorCount)/m))

if __name__=='__main__':
    testRbf(1.4)


