import numpy as np

print('*'*40+'数组、矩阵的性质'+'*'*30)
x=np.random.random_sample((3, 2))#得到3行、2列的数组,"numpy.ndarray"
print(x.shape[0])#行数
print(x.shape[1])#列数
print(x.dtype)#数据类型
print(x.flags)#数组的存储情况
print(x.shape)#每一维的大小
print(x.ndim)#数组的维数
print(x.size)#含有元素的数目
print(x.itemsize)#所占内存大小
print(x.T)#矩阵转置
print('*'*40)

print('*'*30+'矩阵、向量、数组的建立'+'*'*30)
x=np.random.random(5)#随机size个float数位于[0.0,1.0),"numpy.ndarray"
print(x)
print(x+1)
x=np.random.random_sample()#随机得到一个float，位于[0.0,1.0)
print(x)
x=np.random.random_sample((5,))#得到5个,"numpy.ndarray"
print(x)
x=np.random.random_sample((3, 2))#得到3行、2列的数组,"numpy.ndarray"
print(np.zeros((2,3)))#建立2*3的全0矩阵
print(np.ones((2,3)))#建立全1矩阵
print(np.empty((2,3),dtype=int))#建立空矩阵（给定形状、dtype和order的未初始化(任意)数据数组）
print(np.eye(3,2,0))#建立主对角线为1的矩阵
print(np.identity(4))#建立n*n的单位阵
print(np.linspace(1,100,34))#将1到100线性划分为34个点
print(np.logspace(1,100,34,base=10.0))#返回的数字在对数尺度上均匀分布

print('*'*40+'数组、矩阵的操作'+'*'*40)
x=np.random.randn(2,4)#从标准正态分布中返回一个样本
x=np.arange(10)#0~9
x=np.arange(2,10,1)#在[2,10)范围内从1开始步长为1的数组
x=x.reshape(2,4)#重塑为2*4的数组。要求数目相同
print(x)
x=np.array([[1,2,3],[4,5,6]])
print(np.flip(x,1))#在给定的轴上反转数组
print(np.flipud(x))#在上下方向翻转
print(np.rot90(x))#在坐标轴指定的平面旋转90度
x=x.ravel()#重塑和定型的反向操作
print(x,np.roll(x,2))#x循环右移2位
a=np.random.random((3,3))
b=np.random.random((3,1))
print(np.dot(a,b))#矩阵乘法
print(np.vdot(a[0],b.T))#矢量数量积
print(np.inner(a,b.T))#内积，点乘，矩阵乘法
print(np.outer(a[0],b.T))#向量的外积:a = [a0, a1, ..., aM]与b = [b0, b1, ..., bN]，[[a0*b0  a0*b1 ... a0*bN ]
                                                                                   #[a1*b0    .
                                                                                   #[ ...          .
                                                                                   #[aM*b0            aM*bN ]]


