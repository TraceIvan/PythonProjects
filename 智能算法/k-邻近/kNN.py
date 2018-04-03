from numpy import  *
import operator
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=15)

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

#参数：用于分类的输入向量；输入的训练样本集；标签向量；用于选择最近邻居的数目
#标签向量的元素数目和矩阵dataSet行数相同
def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    #距离计算（使用欧式距离公式）
    diffMat=tile(inX,(dataSetSize,1))-dataSet#矩阵对应值相减
    sqDiffMat=diffMat**2#矩阵中各个元素平方
    sqDistances=sqDiffMat.sum(axis=1)#按对应维求和
    distances=sqDistances**0.5#矩阵中各个元素开方
    sortedDistIndicies=distances.argsort()#从小到大排序，输出为对应下标（从0开始）
    classCount={}
    #选择距离最小的k个点
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    #排序
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#将文本记录转换为Numpy
def file2maxtrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)#得到行数
    returnMat=zeros((numberOfLines,3))#创建返回的NumPy矩阵
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()#截取掉所有回车字符
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]#选取前3个特征存储到特征矩阵
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector

#归一化特征
def autoNorm(dataset):
    minVals=dataset.min(axis=0)#按列取最小
    maxVals=dataset.max(axis=0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataset))
    m=dataset.shape[0]
    normDataSet=dataset-tile(minVals,(m,1))
    normDataSet=dataset/tile(ranges,(m,1))#对应元素相除
    return normDataSet,ranges,minVals,maxVals


if __name__=='__main__':
    group, labels = createDataSet()
    print(classify0([0.7,1],group,labels,3))

    #该数据集中第一、二、三列为特征，最后一列为类别。每行代表一个数据
    filename='datingTestSet2.txt'
    featureMat,Labels=file2maxtrix(filename)
    featureMat,ranges,minVals,maxVals=autoNorm(featureMat)
    print(featureMat)

    # 将三类数据分别取出来
    # x轴代表飞行的里程数
    # y轴代表玩视频游戏的百分比
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    for i in range(len(Labels)):
        if Labels[i] == 1:  # 不喜欢
            type1_x.append(featureMat[i][0])
            type1_y.append(featureMat[i][1])

        if Labels[i] == 2:  # 魅力一般
            type2_x.append(featureMat[i][0])
            type2_y.append(featureMat[i][1])

        if Labels[i] == 3:  # 极具魅力
            type3_x.append(featureMat[i][0])
            type3_y.append(featureMat[i][1])
    plt.figure(1)
    axes=plt.subplot(111)
    type1 = axes.scatter(type1_x, type1_y, s=20, c='red')
    type2 = axes.scatter(type2_x, type2_y, s=40, c='green')
    type3 = axes.scatter(type3_x, type3_y, s=50, c='blue')
    # plt.scatter(matrix[:, 0], matrix[:, 1], s=20 * numpy.array(labels),
    #             c=50 * numpy.array(labels), marker='o',
    #             label='test')
    plt.xlabel('每年获取的飞行里程数')
    plt.ylabel('玩视频游戏所消耗的事件百分比')
    axes.legend((type1, type2, type3), ('不喜欢', '魅力一般', '极具魅力'))

    plt.show()

    ratio_of_test=0.1
    numTest=int(ratio_of_test*featureMat.shape[0])
    error_count=0
    #将前numTest作为测试样例
    for i in range(numTest):
        classify_result=classify0(featureMat[i,:],featureMat[numTest:featureMat.shape[0],:],Labels[numTest:],3)
        print("the predicted answer is %d, the right answer is %d."%(classify_result,Labels[i]))
        if classify_result!=Labels[i]:
            error_count+=1
    print('the error rate is %f.'%(1.0*error_count/numTest))