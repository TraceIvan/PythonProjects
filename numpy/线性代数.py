import numpy as np
a=np.random.random((2,3))
print(a)
b=np.random.random((2,3))
print(b)
c=np.random.random((3,2))
print(c)
d=np.random.random((2,1))
print(d)
e=np.abs(np.random.random((3,3)))
print(e)
r1=np.dot(c,a)#点乘，矩阵乘法
print(r1)
print(np.linalg.norm(a))#矩阵或向量的模--范数
print(np.linalg.cond(a))#矩阵的条件数
print(np.linalg.solve(e,r1))#解Ax=b，A必须为方阵
print(np.linalg.det(e))#行列式
print(np.linalg.inv(e))#逆矩阵
print(np.linalg.pinv(e))#A的伪逆矩阵
print(np.linalg.eig(e))#A的特征值和向量
print(np.linalg.eigvals(e))#A的特征根
print(np.linalg.svd(e))#奇异值分解
t1=np.array([[4,12,-16],
             [12,37,-43],
             [-16,-43,98]])
print(np.linalg.cholesky(t1))#柯列斯基分解:将正定矩阵分解为上三角矩阵和下三角矩阵
print(np.linalg.qr(e))#QR分解

