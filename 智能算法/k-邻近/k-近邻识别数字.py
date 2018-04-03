import tensorflow as tf
import numpy as np
import os
import kNN
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#将文本格式的32*32黑白图像转换为1*1024的向量
def imgtxt2vector(filename):
    returnVector=np.zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVector[0,32*i+j]=int(lineStr[j])
    return returnVector

def handwriteringClassTest():
    hwLabels=[]
    trainingFileList=os.listdir('digits/trainingDigits')#获取目录内容
    m=len(trainingFileList)
    trainingMat=np.zeros([m,1024])
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=imgtxt2vector('digits/trainingDigits/'+fileNameStr)
    testFileList=os.listdir('digits/testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=imgtxt2vector('digits/testDigits/'+fileNameStr)
        classifierResult=kNN.classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print('the predicted answer is %d,the true answer is %d.'%(classifierResult,classNumStr))
        if classifierResult!=classNumStr:
            errorCount+=1
    print('the total error rate is %f.'%(errorCount/mTest))
    
if __name__=='__main__':
    handwriteringClassTest()




