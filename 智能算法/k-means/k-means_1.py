#!/usr/bin/python
# coding=utf-8
import numpy as np
# 加载数据
def loadDataSet(fileName):  # 解析文件，按tab分割字段，得到一个浮点数字类型的矩阵
    dataMat = []              # 文件的最后一个字段是类别标签
    data_y=[]
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split()
        fltLine = list(map(float, curLine))# 将每个元素转成float类型
        data_y.append(int(fltLine[-1]))
        dataMat.append(fltLine[:-1])
    return dataMat,data_y

# 计算欧几里得距离
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2))) # 求两个向量之间的距离

# 构建聚簇中心，取k个随机质心
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))   # 每个质心有n个坐标值，总共要k个质心
    for j in range(n):
        minJ = np.min(dataSet[:,j])
        maxJ = np.max(dataSet[:,j])
        rangeJ = float(maxJ - minJ)
        centroids[:,j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids

# k-means 聚类算法
def kMeans(dataSet, k, distMeans =distEclud, createCent = randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))    # 用于存放该样本属于哪类及质心距离
    # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
    centroids = createCent(dataSet, k)
    clusterChanged = True   # 用来判断聚类是否已经收敛
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 把每一个数据点划分到离它最近的中心点
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeans(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
            if clusterAssment[i,0] != minIndex: clusterChanged = True;  # 如果分配发生变化，则需要继续迭代
            clusterAssment[i,:] = minIndex,minDist**2   # 并将第i个数据点的分配情况存入字典
        #print("簇中心：",centroids)
        for cent in range(k):   # 重新计算中心点
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A == cent)[0]]   # 去第一列等于cent的所有列
            centroids[cent,:] = np.mean(ptsInClust, axis = 0)  # 算出这些数据的中心点
    return centroids, clusterAssment

# 二分K均值聚类的主函数，输入参数为数据集和要分的簇的个数
def bi_Kmeans(data_set, k):
    m = data_set.shape[0]   # 得到data_set的行数，即数据集的个数
    cluster_assment = np.zeros((m,2)) # 初始化样本点的簇分配矩阵，第一列为簇分配索引，第二列为欧氏距离平方
    centroid0 = np.mean(data_set, axis=0)  # 按列计算均值，即找到初始质心
    cent_list = [centroid0]
    for j in range(m):   # 对于每个样本点
        cluster_assment[j,1] = distEclud(centroid0, data_set[j,:])**2 # 计算该样本点的误差平方
    while (len(cent_list) < k):   # 当已有的簇个数小于k时，迭代执行以下代码
        lowestSSE = np.inf     # 初始化误差平方和SSE的最小值

        # 找到对哪个簇进行划分可以最大程度降低SSE值
        for i in range(len(cent_list)):   # 遍历每个已有的簇
            # 得到属于该簇的所有样本点
            pts_incurr_cluster = \
            data_set[np.nonzero(list(map(lambda x:x==i, cluster_assment[:,0])))]
            # 将该簇的所有样本点通过函数Kmean进行划分(k=2),得到划分后的质心和簇分配矩阵
            centroid_mat, split_clust_ass = kMeans(pts_incurr_cluster, 2)
            sse_split = np.sum(split_clust_ass[:,1])  # 得到划分后的误差平方和
            # 得到其他样本点的误差平方和
            sse_not_split = \
            np.sum(cluster_assment[np.nonzero(list(map(lambda x:x!=i, cluster_assment[:,0]))),1])
            if (sse_split + sse_not_split) < lowestSSE:  # 如果总的误差平方和小于lowestSSE
                best_cent_to_split = i                   # 则保存本次划分
                best_new_cents = centroid_mat
                best_clust_ass = split_clust_ass
                lowestSSE = sse_split + sse_not_split
        # 对最大程度降低SSE值的簇进行划分

        # 将划分后得到的编号为0的结果簇的编号修改为原最大簇编号+1，即len(cent_list)
        best_clust_ass[np.nonzero(list(map(lambda x:x==1, best_clust_ass[:,0]))), 0] = len(cent_list)
         # 将划分后得到的编号为1的结果簇的编号修改为被划分的簇的编号
        best_clust_ass[np.nonzero(list(map(lambda x:x==0, best_clust_ass[:,0]))), 0] = best_cent_to_split
        cent_list[best_cent_to_split] = best_new_cents[0,:] # 更新被划分的簇的质心
        cent_list.append(best_new_cents[1,:])  # 添加一个新的簇的质心
        cluster_assment[np.nonzero(list(map(lambda x:x==best_cent_to_split, cluster_assment[:,0]))),:] = \
            best_clust_ass  # 将簇分配矩阵中属于被划分的簇的样本点的簇分配进行更新
    return np.array(cent_list), cluster_assment

def show(dataSet, k, centroids, clusterAssment):
    from matplotlib import pyplot as plt
    numSamples, dim = dataSet.shape
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
    plt.show()

# --------------------测试----------------------------------------------------
# 用测试数据及测试kmeans算法
def main():
    dataMat,data_Y=loadDataSet('seeds_dataset.txt')
    dataMat=np.mat(dataMat)
    myCentroids, clustAssing = kMeans(dataMat, 3)
    print("簇中心：",myCentroids)
    #print("样本所属簇以及距离簇中心距离\n",clustAssing)
    print("距离簇中心距离总和：",clustAssing.sum(axis=0)[0,1])
    labels=clustAssing[:,0].flatten().tolist()[0]
    labels=list(map(int,labels))
    conf_mul = np.zeros(shape=(3, 3))
    for idx, val in enumerate(labels):
        conf_mul[int(data_Y[idx]) - 1, val] += 1
    print("混淆矩阵\n",conf_mul)
    maxs = conf_mul.max(axis=0)  # 列最大值
    sums = conf_mul.sum(axis=0)  # 列和
    subs = sums - maxs
    errors = np.sum(subs)
    err_rate = np.sum(subs) / np.sum(sums)
    print("错误数：", errors)
    print("错误率：", err_rate)

    #show(dataMat, 4, myCentroids, clustAssing)


if __name__ == '__main__':
    main()
