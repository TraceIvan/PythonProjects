import numpy as np
import random

'''装载数据'''
def load():
    data=np.loadtxt('data\k-means.csv',delimiter=',')
    return data

'''计算距离'''
def calcDis(data,clu,k):
    clalist=[]  #存放计算距离后的list
    data=data.tolist()  #转化为列表
    clu=clu.tolist()
    for i in range(len(data)):
        clalist.append([])
        for j in range(k):
            dist=round(((data[i][1]-clu[j][0])**2+(data[i][2]-clu[j][1])**2)*0.05,1)
            clalist[i].append(dist)
    clalist=np.array(clalist)   #转化为数组
    return clalist

'''分组'''
def group(data,clalist,k):
    grouplist=[]    #存放分组后的集群
    claList=clalist.tolist()
    data=data.tolist()
    for i in range(k):
        #确定要分组的个数，以空列表的形式，方便下面进行数据的插入
        grouplist.append([])
    for j in range(len(clalist)):
        sortNum=np.argsort(clalist[j])
        grouplist[sortNum[0]].append(data[j][1:])
    grouplist=np.array(grouplist)
    return grouplist

'''计算质心'''
def calcCen(data,grouplist,k):
    clunew=[]
    data=data.tolist()
    grouplist=grouplist.tolist()
    templist=[]
    #templist=np.array(templist)
    for i in range(k):
        #计算每个组的新质心
        sumx=0
        sumy=0
        for j in range(len(grouplist[i])):
            sumx+=grouplist[i][j][0]
            sumy+=grouplist[i][j][1]
        clunew.append([round(sumx/len(grouplist[i]),1),round(sumy/len(grouplist[i]),1)])
    clunew=np.array(clunew)
    #clunew=np.mean(grouplist,axis=1)
    return clunew

'''优化质心'''
def classify(data,clu,k):
    clalist=calcDis(data,clu,k) #计算样本到质心的距离
    grouplist=group(data,clalist,k) #分组
    for i in range(k):
        #替换空值
        if grouplist[i]==[]:
            grouplist[i]=[4838.9,1926.1]
    clunew=calcCen(data,grouplist,k)
    sse=clunew-clu
    #print "the clu is :%r\nthe group is :%r\nthe clunew is :%r\nthe sse is :%r" %(clu,grouplist,clunew,sse)
    return sse,clunew,data,k

if __name__=='__main__':
    k=3 #给出要分类的个数的k值
    data=load() #装载数据
    clu=random.sample(data[:,1:].tolist(),k)    #随机取质心
    clu=np.array(clu)
    sse,clunew,data,k=classify(data,clu,k)
    while np.any(sse!=0):
        sse,clunew,data,k=classify(data,clunew,k)
    clunew=np.sort(clunew,axis=0)
    print("the best cluster is %r" %clunew)