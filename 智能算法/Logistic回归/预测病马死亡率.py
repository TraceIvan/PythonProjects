import logRegres_bin as LG
import numpy as np
def classifyVector(inX,weights):
    x=(np.dot(inX,weights))
    prob=LG.sigmoid(x[0,0])
    if prob>0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    trainingSet=[];trainingLabels=[]
    with open('horseColicTraining.txt','rb') as fr:
        for line in fr.readlines():
            curLine=line.decode('utf8').strip().split('\t')
            linArr=[]
            for i in range(21):
                linArr.append(float(curLine[i]))
            trainingSet.append(linArr)
            trainingLabels.append(float(curLine[21].strip()))

    trainWeights,_,_,_=LG.stocGradAscent0(np.mat(trainingSet),np.mat(trainingLabels),500)
    errorCount=0;numTestVec=0.0
    with open('horseColicTest.txt','rb') as fr:
        for line in fr.readlines():
            numTestVec+=1
            curLine=line.decode('utf8').strip().split('\t')
            lineArr=[]
            for i in range(21):
                lineArr.append(float(curLine[i].strip()))
            if int(classifyVector(np.mat(lineArr),weights=trainWeights))!=int(curLine[21]):
                errorCount+=1
    errorRate=float(errorCount/numTestVec)
    print('the error rate of this test is : %f.'%errorRate)
    return trainWeights,errorRate

def multiTest():
    numTests=10;errorSum=0.0
    for k in range(numTests):
        _,curError=colicTest()
        errorSum+=curError
    print('after %d iterations the average error rate is: %f.'%(numTests,errorSum/numTests))

if __name__=='__main__':
    multiTest()




