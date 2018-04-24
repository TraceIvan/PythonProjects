'''
本模块将通过OVA和AVA(一对多和多对多进行分类)，对手写数字(10个类别)进行投票
'''
import os
import numpy as np
import  random
class LR_OVA(object):
    def __init__(self):
        self.dataMat,self.labelMat=self.loadData('digits/trainingDigits')
        self.testMat,self.testLabelMat=self.loadData('digits/testDigits')
        self.classifyArr=[]
        self.OVATraining()
        self.errRate=self.OVATest()
    def img2vector(self,TextDir):
        imgVector=[]
        with open(TextDir,'r',encoding='utf8') as fr:
            for line in fr.readlines():
                length=len(line)
                for i in line.strip():
                    imgVector.append(int(i))
        imgVector=np.mat(imgVector)
        return imgVector

    def loadData(self,fileDir):
        fileList=os.listdir(fileDir)
        m=len(fileList)
        dataMat=np.zeros((m,1024))
        labelMat=np.zeros((m,1))
        for i in range(m):
            fileName=fileList[i].split('.')[0]
            fileLabel=fileName.split('_')[0]
            labelMat[i,0]=int(fileLabel)
            dataMat[i,:]=self.img2vector(fileDir+'/'+fileList[i])
        dataMat=np.mat(dataMat)
        labelMat=np.mat(labelMat)
        return dataMat,labelMat

    # 梯度上升法
    def gradAscent(self,dataMat, labelMat):
        m, n = np.shape(dataMat)
        alpha = 0.001  # 步长
        maxCycles = 1000  # 迭代次数
        weights = np.ones((n, 1))
        weights=np.mat(weights)
        for k in range(maxCycles):
            h = self.sigmoid(np.dot(dataMat , weights))
            error = (labelMat - h)  # 计算真实类别与预测类别的差值
            weights = weights + alpha * dataMat.transpose() * error
        return weights

    # 随机梯度上升
    def stocGradAscent0(self,dataMatrix, classLabels, numIter=50):
        '''输入均为mat'''
        m, n = np.shape(dataMatrix)
        weights = np.ones([n, 1])
        for j in range(numIter):
            for i in range(m):
                alpha = 0.0001 + 10/ (1.0 + i + j)  # alpha每次迭代时进行调整
                randIndex = int(random.uniform(0, m))  # 随机选择一个样例进行计算来更新回归系数
                g = self.sigmoid(dataMatrix[randIndex] * weights)
                error = classLabels[randIndex,0] - g
                weights = weights + dataMatrix[randIndex].transpose() * alpha * error
                weights.ravel()
        return weights
    # sigmoid函数
    def sigmoid(self,inX):
        return np.longfloat(1.0 / (1 + np.exp(-inX)))

    def classify(self,inx,weights):
        classified=self.sigmoid(inx*weights)
        if classified>0.5:
            return 1
        else:
            return 0

    def OVATraining(self):
        for i in range(10):
            dataMat=self.dataMat.copy()
            m=np.shape(self.labelMat)[0]
            labelMat=np.zeros((m,1))
            labelMat=np.mat(labelMat)
            for j in range(m):
                if self.labelMat[j]==i:
                    labelMat[j]=1
            weights=self.stocGradAscent0(dataMat,labelMat)
            self.classifyArr.append(weights)

    def OVATest(self):
        m=np.shape(self.testLabelMat)[0]
        errCnt=0
        for i in range(m):
            inX=self.testMat[i]
            ticketCount=np.zeros((1,10))
            for j in range(10):
                predicted=self.classify(inX,self.classifyArr[j])
                if predicted==1:
                    ticketCount[0,j]+=1
                else:
                    ticketCount[0,j]-=1
            predicted=ticketCount.argmax()
            if predicted!=self.testLabelMat[i,0]:
                errCnt+=1
        errRate=errCnt/m*100
        print('the error rate is %.2f%%.'%errRate)
        return errRate


if __name__=='__main__':
    LR_OVA()


