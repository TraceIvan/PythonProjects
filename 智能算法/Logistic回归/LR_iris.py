import numpy as np
def loadData(fileName):
    dataArr=[]
    classArr=[]
    with open(fileName,'r',encoding='utf8') as fr:
        for line in fr.readlines():
            lineArr=line.strip().split(',')
            dataArr.append(list(map(float,lineArr[:-1])))
            classArr.append(lineArr[-1])
    return dataArr,classArr

#归一化特征
def autoNorm(dataset):
    minVals=dataset.min(axis=0)#按列取最小
    maxVals=dataset.max(axis=0)
    ranges=maxVals-minVals
    normDataSet=np.zeros(np.shape(dataset))
    m=dataset.shape[0]
    normDataSet=dataset-np.tile(minVals,(m,1))
    normDataSet=dataset/np.tile(ranges,(m,1))#对应元素相除
    return normDataSet,ranges,minVals,maxVals

if __name__=='__main__':
    dataArr,classArr=loadData('iris.data')
    labels = ['sepal length', 'sepal width', 'petal length', 'petal width']
    dataMat=np.mat(dataArr)
    classArr=
