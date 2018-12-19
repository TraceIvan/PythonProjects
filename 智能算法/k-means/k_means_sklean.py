import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def getData():
    # 准备数据集
    data_path='seeds_dataset.csv'#一个具有四种类别，两种不同features(特征)的小数据集
    df=pd.read_csv(data_path)
    print(df)
    print(df.head())
    print(df.info()) # 查看数据信息，确保没有错误
    dataset_X,dataset_y=df.iloc[:,:-1],df.iloc[:,-1]#切片
    print(dataset_X.head())
    print(dataset_X.info())
    #print('-'*100)
    #print(dataset_y.head())
    dataset_X=dataset_X.values
    dataset_y=dataset_y.values
    return dataset_X,dataset_y
    # print(dataset_X.shape) # (400, 2)
    # print(dataset_y.shape) # (400,)


# 无标签数据集可视化，将第一列feature作为X，第二列feature作为y
def visual_2D_dataset_dist(dataset):
    '''将二维数据集dataset显示在散点图中'''
    assert dataset.shape[1] == 2, 'only support dataset with 2 features'
    plt.figure()
    X = dataset[:, 0]
    Y = dataset[:, 1]
    plt.scatter(X, Y, marker='v', c='g', label='dataset')

    X_min, X_max = np.min(X) - 1, np.max(X) + 1
    Y_min, Y_max = np.min(Y) - 1, np.max(Y) + 1
    plt.title('dataset distribution')
    plt.xlim(X_min, X_max)
    plt.ylim(Y_min, Y_max)
    plt.xlabel('feature_0')
    plt.ylabel('feature_1')
    plt.legend()
    plt.show()


#可视化K-means聚类结果
def visual_kmeans_effect(k_means, dataset):
    assert dataset.shape[1] == 2, 'only support dataset with 2 features'
    X = dataset[:, 0]
    Y = dataset[:, 1]
    X_min, X_max = np.min(X) - 1, np.max(X) + 1
    Y_min, Y_max = np.min(Y) - 1, np.max(Y) + 1
    X_values, Y_values = np.meshgrid(np.arange(X_min, X_max, 0.01),
                                     np.arange(Y_min, Y_max, 0.01))
    # 预测网格点的标记
    predict_labels = k_means.predict(np.c_[X_values.ravel(), Y_values.ravel()])
    predict_labels = predict_labels.reshape(X_values.shape)
    plt.figure()
    plt.imshow(predict_labels, interpolation='nearest',
               extent=(X_values.min(), X_values.max(),
                       Y_values.min(), Y_values.max()),
               cmap=plt.cm.Paired,
               aspect='auto',
               origin='lower')

    # 将数据集绘制到图表中
    plt.scatter(X, Y, marker='v', facecolors='none', edgecolors='k', s=30)

    # 将中心点回执到图中
    centroids = k_means.cluster_centers_#中心点
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o',
                s=100, linewidths=2, color='k', zorder=5, facecolors='b')
    plt.title('K-Means effect graph')
    plt.xlim(X_min, X_max)
    plt.ylim(Y_min, Y_max)
    plt.xlabel('feature_0')
    plt.ylabel('feature_1')
    plt.show()

def K_means(dataset_X,dataset_y,n_clusters):
    # 定义一个k-means对象
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    # 这几个参数是初始化设定的，其中n_clusters是从二维散点图中看出大概有4个族群
    kmeans.fit(dataset_X)
    label_pred = kmeans.labels_  # 获取聚类标签
    print("聚类标签\n",label_pred)
    centroids = kmeans.cluster_centers_  # 获取聚类中心
    print("聚类中心\n",centroids)
    inertia = kmeans.inertia_  # 获取聚类准则的总和
    print("聚类准则的总和\n",inertia)
    conf_mul=np.zeros(shape=(3,3))
    for idx,val in enumerate(label_pred):
        conf_mul[int(dataset_y[idx])-1,val]+=1
    print("混淆矩阵\n", conf_mul)
    maxs =conf_mul.max(axis=0)#列最大值
    sums=conf_mul.sum(axis=0)#列和
    subs=sums-maxs
    errors=np.sum(subs)
    err_rate=np.sum(subs)/np.sum(sums)
    print("错误数：",errors)
    print("错误率：",err_rate)
    return kmeans

def K_means_folds(dataset_X,dataset_y,n_clusters,folds):
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    maxn,cols=np.shape(dataset_X)
    siz=maxn//folds
    for i in range(folds):
        curl=i*siz
        curr=(i+1)*siz
        if i==folds-1 :curr=maxn
        test_X=dataset_X[curl:curr]
        test_Y=dataset_y[curl:curr]
        train_X=np.vstack((dataset_X[0:curl],dataset_X[curr:maxn]))
        kmeans.fit(train_X)
        labels=kmeans.labels_
        conf_mul = np.zeros(shape=(n_clusters, n_clusters))
        for idx, val in enumerate(labels):
            conf_mul[int(dataset_y[idx]) - 1, val] += 1
        mp=conf_mul.argmax(axis=0)
        pre=kmeans.predict(test_X)
        pre=[mp[i]+1 for i in pre]
        err=0
        for j in range(len(pre)):
            if pre[j]!=test_Y[j]: err+=1
        print(i,":errs:",err,",err_rate:",err/len(pre))


if __name__=='__main__':
    dataset_x,dataset_y=getData()
    print(dataset_x)
    print(dataset_y)
    #visual_2D_dataset_dist(dataset_x)
    k_means_ret=K_means(dataset_x,dataset_y,3)
    #visual_kmeans_effect(k_means_ret,dataset_x)
    K_means_folds(dataset_x,dataset_y,3,10)




