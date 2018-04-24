import matplotlib.pyplot as plt
import numpy as np
def plotROC(predStrengths,classLabels):
    '''
    :param predStrengths:分类器的预测强度
    朴素贝叶斯中的可能性、Logistic回归输到Sigmoid函数的数值、AdaBoost与SVM输入到sign()函数的数值
    :param classLabels: 类别标签
    :return:
    '''
    cur=(1.0,1.0)#绘制光标的位置
    ySum=0.0#用于计算AUC(area under the curve)
    numPosClas=sum(np.array(classLabels)==1.0)#通过数组过滤计算正例的数目
    yStep=1/float(numPosClas)#y轴上的步进数目
    xStep=1/float(len(classLabels)-numPosClas)
    sortedIndicies=predStrengths.argsort()#h获取排好序的索引，从最小到最大，需要从(1.0,1.0)绘制到(0.0,0.0)
    fig=plt.figure()
    fig.clf()
    ax=plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:#在所有排序值上循环
        if classLabels[index]==1.0:#每得到一个标签为1的类，沿y轴下降一个步长,降低假阳率
            delX=0
            delY=yStep
        else:#否则x轴上倒退一个步长，降低假阴率
            delX=xStep
            delY=0
            ySum+=cur[1]#累加多个小矩形的面积(由于宽度相同，先累加高度)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur=(cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    ax.axis([0,1,0,1])
    plt.show()
    print('the Area Under the Curve is: ',ySum*xStep)
