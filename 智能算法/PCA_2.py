import numpy as np
from sklearn.decomposition import PCA

def pca(X,k):#k is the components you want
  #mean of each feature 求每个特征的平均值
  n_samples, n_features = X.shape
  mean=np.array([np.mean(X[:,i]) for i in range(n_features)])
  #normalization 标准化，减去平均值
  norm_X=X-mean
  #scatter matrix 散布矩阵（离差矩阵）——协方差矩阵
  scatter_matrix=np.dot(np.transpose(norm_X),norm_X)
  #Calculate the eigenvectors and eigenvalues 求其特征值和特征向量
  eig_val, eig_vec = np.linalg.eig(scatter_matrix)
  eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
  # sort eig_vec based on eig_val from highest to lowest 根据特征值从大到小排序取特征向量
  eig_pairs.sort(reverse=True)
  # select the top k eig_vec 选前k个特征向量
  feature=np.array([ele[1] for ele in eig_pairs[:k]])
  #get new data 得到降维后的矩阵
  data=np.dot(norm_X,np.transpose(feature))
  return data

if __name__=='__main__':
    X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    result1=pca(X,1)
    print(result1)

    rst2=PCA(n_components=1)
    rst2.fit(X)
    result2=rst2.transform(X)
    print(result2)

