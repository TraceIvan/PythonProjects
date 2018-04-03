import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass

    def train(self,X,y):
        '''X是N*D，每一行为一个测试示例。y是一维的，大小为N'''
        #O(1)
        self.Xtr=X
        self.ytr=y

    def predict(self,X):
        '''X大小为N*D，每一行为所要预测的数据'''
        num_test=X.shape[0]
        #确保输出类型和输入类型一样
        Ypred=np.zeros(num_test,dtype=self.ytr.dtype)

        #通过循环来预测
        for i in range(num_test):
            #计算L1距离，找到最近邻
            distances=np.sum(np.abs(self.Xtr-X[i,:]),axis=1)#按行求和
            min_index=np.argmin(distances)#取最小
            Ypred[i]=self.ytr[min_index]

        return Ypred

if __name__=='__main__':
    tx=np.array([[1,2,3,4],
        [11,12,13,14]])
    px=np.array([100,101,102,103])
    distance=np.sum(np.abs(tx-px),axis=0)
    print(distance)