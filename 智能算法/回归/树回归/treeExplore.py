import tkinter as tk
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')#设定matplotlib后端为TkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
########################回归树和模型树########################
#载入数据，最后一列为y值
def loadDataSet(fileName):
    dataMat=[]
    with open(fileName,'r',encoding='utf8') as fr:
        lines=fr.readlines()
        for line in lines:
            curLine=line.strip().split()
            fltLine=list(map(float,curLine))#将每行映射为浮点数
            dataMat.append(fltLine)
    return np.mat(dataMat)
#按照指定特征及其阈值将数据集划分为2个子集
def binSplitDataSet(dataSet,feature,value):
    '''
    :param dataSet: 数据集合
    :param feature: 待切分的特征
    :param value: 该特征的某个值
    :return: 两个子集
    '''
    mat0=dataSet[np.nonzero(dataSet[:,feature]>value)[0],:]
    mat1=dataSet[np.nonzero(dataSet[:,feature]<=value)[0],:]
    return mat0,mat1
#生成叶结点
def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])
#误差估计函数，计算平方误差总和
def regErr(dataSet):
    return np.var(dataSet[:,-1])*dataSet.shape[0]
#对数据集进行OLS线性回归
def linearSolve(dataMat):
    m,n=dataMat.shape
    X,Y=np.mat(np.ones((m,n))),np.mat(np.ones((m,1)))
    X[:,1:n]=dataMat[:,0:n-1]#第一列为常数项
    Y=dataMat[:,-1]
    xTx=np.dot(X.T,X)
    if np.linalg.det(xTx)==0.0:
        raise NameError("This matrix is singular, can't do inverse.\nTry increasing the second value of ops.")
    ws=np.dot(xTx.I,np.dot(X.T,Y))
    return ws,X,Y
#模型树建立叶结点
def modelLeaf(dataMat):
    ws,X,Y=linearSolve(dataMat)
    return ws
#模型树误差计算
def modelErr(dataMat):
    ws,X,Y=linearSolve(dataMat)
    yHat=np.dot(X,ws)
    return np.sum(np.power(Y-yHat,2))
#找到最佳的二元切分方式
def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS,tolN=ops[0],ops[1]#自定义容许的误差下降值、切分的最少样本数
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:#剩余样本y取值只有1种情况，无需切分
        return None,leafType(dataSet)
    m,n=dataSet.shape
    S=errType(dataSet)#当前数据集的总平方误差
    bestS,bestIndex,bestValue=np.inf,0,0
    #在所有可能的特征及其可能取值上遍历
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].flatten().A[0]):
            mat0,mat1=binSplitDataSet(dataSet,featIndex,splitVal)
            if mat0.shape[0]<tolN or mat1.shape[0]<tolN:
                continue
            newS=errType(mat0)+errType(mat1)
            if newS<bestS:
                bestS=newS
                bestIndex=featIndex
                bestValue=splitVal

    if (S-bestS)<tolS:#误差下降不明显，则直接创建叶结点
        return None,leafType(dataSet)
    mat0,mat1=binSplitDataSet(dataSet,bestIndex,bestValue)
    if mat0.shape[0] < tolN or mat1.shape[0] < tolN:#如果某个子集的大小小于切分最小样本数，则也不进行切分
        return None,leafType(dataSet)
    return bestIndex,bestValue#返回特征与阈值

#建立树
def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    '''
    :param dataSet: 数据集
    :param leafType: 建立叶结点的函数
    :param errType: 误差计算函数
    :param ops: 树构建所需其他参数的元组
    :return: 树
    '''
    feat,val=chooseBestSplit(dataSet,leafType,errType,ops)
    if feat==None:
        return val
    retTree={}
    retTree['spInd']=feat
    retTree['spVal']=val
    lSet,rSet=binSplitDataSet(dataSet,feat,val)
    retTree['left']=createTree(lSet,leafType,errType,ops)
    retTree['right']=createTree(rSet,leafType,errType,ops)
    return retTree
#判断是否为树
def isTree(obj):
    return (type(obj).__name__=='dict')
#对树进行塌陷处理（返回树平均值）
def getMean(tree):
    if isTree(tree['right']):
        tree['right']=getMean(tree['right'])
    if isTree(tree['left']):
        tree['left']=getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
#对树进行后剪枝
def prune(tree,testData):
    '''
    :param tree: 待剪枝的树
    :param testData: 用于剪枝的测试数据
    :return:
    '''
    if testData.shape[0]==0:#如果测试集为空
        return getMean(tree)
    lSet,rSet=[],[]
    if (isTree(tree['left'])) or (isTree(tree['right'])):
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):#对左子树剪枝
        tree['left']=prune(tree['left'],lSet)
    if isTree(tree['right']):#对右子树剪枝
        tree['right']=prune(tree['right'],rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):#如果左右分支均非树结构，考虑合并
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge=sum(np.power(lSet[:,-1]-tree['left'],2))+sum(np.power(rSet[:,-1]-tree['right'],2))
        treeMean=(tree['left']+tree['right'])/2.0
        errorMerge=sum(np.power(testData[:,-1]-treeMean,2))
        if errorMerge<errorNoMerge:#如果叶结点合并能够降低测试误差，则合并
            print('Merging')
            return treeMean
        else:
            return tree
    else:
        return tree

#回归树叶结点预测
def regTreeEval(model,inDat):
    return float(model)
# 模型树叶结点预测
def modelTreeEval(model, inDat):
    n=inDat.shape[1]
    X=np.mat(np.ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(np.dot(X,model))
#对某一条测试数据预测
def treeForeCast(tree,inDat,modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree,inDat)
    if inDat[tree['spInd']]>tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inDat,modelEval)
        else:
            return modelEval(tree['left'],inDat)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inDat,modelEval)
        else:
            return modelEval(tree['right'],inDat)

#树预测
def createForeCast(tree,testData,modelEval=regTreeEval):
    m=len(testData)
    yHat=np.mat(np.ones((m,1)))
    for i in range(m):
        yHat[i,0]=treeForeCast(tree,np.mat(testData[i]),modelEval)
    return yHat

#相关系数评价
def corrCoef(yHat,testMat):
    return np.corrcoef(yHat,testMat[:,-1])
########################结束########################
def reDraw(tolS,tolN):
    reDraw.f.clf()#清空之前的图像
    reDraw.a=reDraw.f.add_subplot(111)#添加新图
    if chkBtnVal.get():#建立模型树
        if tolN<2:
            tolN=2
        myTree=createTree(reDraw.rawDat,modelLeaf,modelErr,(tolS,tolN))
        yHat=createForeCast(myTree,reDraw.testDat,modelTreeEval)
    else:#建立回归树
        myTree = createTree(reDraw.rawDat, regLeaf, regErr, (tolS, tolN))
        yHat = createForeCast(myTree, reDraw.testDat, regTreeEval)
    reDraw.a.scatter(reDraw.rawDat[:,0].flatten().A[0],reDraw.rawDat[:,1].flatten().A[0],s=5)#真实值采用散点图
    reDraw.a.plot(reDraw.testDat,yHat,linewidth=2.0)#预测值采用曲线
    reDraw.canvas.show()

def getInputs():
    try:
        tolN=int(tolNentry.get())
    except:#如果不能把输入文本解析为整数，则清空输入框且恢复默认值
        tolN=10
        print("Enter Integer for tolN")
        tolNentry.delete(0,tk.END)
        tolNentry.insert(0,'10')
    try:
        tolS=float(tolSentry.get())
    except:
        tolS=1.0
        print("Enter float for tolS")
        tolSentry.delete(0,tk.END)
        tolSentry.insert(0,'1.0')
    return tolN,tolS

def drawNewTree():
    tolN,tolS=getInputs()#得到输入框的值
    reDraw(tolS,tolN)#新建图

if __name__=='__main__':
    root=tk.Tk()

    tk.Label(root,text="Plot Place Holder").grid(row=0,columnspan=3)#新建一个标签对象，并设定位置和跨列
    tk.Label(root,text="tolN").grid(row=1,column=0)
    tolNentry=tk.Entry(root)#新建一个文本输入框（允许单行文本输入）
    tolNentry.grid(row=1,column=1)
    tolNentry.insert(0,'10')
    tk.Label(root,text="tolS").grid(row=2,column=0)
    tolSentry=tk.Entry(root)
    tolSentry.grid(row=2,column=1)
    tolSentry.insert(0,'1.0')
    tk.Button(root,text="ReDraw",command=drawNewTree).grid(row=1,column=2,rowspan=3)#新建一个按钮对象，并设定位置和跨行
    chkBtnVal=tk.IntVar()#新建一个按钮整数值，作为复选按钮的状态变量
    chkBtn=tk.Checkbutton(root,text="Model Tree",variable=chkBtnVal)#新建一个复选按钮
    chkBtn.grid(row=3,column=0,columnspan=2)
    tk.Button(root,text='Quit',fg='black',command=root.quit).grid(row=1,column=2)
    #初始化与reDraw相关的全局变量
    reDraw.rawDat=loadDataSet('sine.txt')
    reDraw.testDat=np.arange(np.min(reDraw.rawDat[:,0]),np.max(reDraw.rawDat[:,0]),0.01)
    reDraw.f=Figure(figsize=(5,4),dpi=100)
    reDraw.canvas=FigureCanvasTkAgg(reDraw.f,master=root)
    reDraw.canvas.show()
    reDraw.canvas.get_tk_widget().grid(row=0,columnspan=3)
    reDraw(1.0,10)

    root.mainloop()#启动事件循环
