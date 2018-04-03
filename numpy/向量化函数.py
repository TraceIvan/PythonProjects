import numpy as np
#第一个例子
def addition(x,y,z):
    return x+y+z

def addpoly():
    i=np.random.randint(25)#返回从[0,25)的一个随机整数
    poly1=np.arange(i,i+10)
    i=np.random.randint(25)
    poly2=np.arange(i,i+10)
    poly3=np.arange(10,20)
    vecf=np.vectorize(addition)#向量化函数,会对每一个元素执行相同的运算
    print(poly1)
    print(poly2)
    print(poly3)
    print('-'*30)
    print(vecf(poly1,poly2,poly3))
addpoly()
#第二个例子
def posquare(x):
    if x>=0:
        return x**2
    else:
        return -x
i=np.random.randint(25)
poly1=np.arange(i,i+10)
vecfunc=np.vectorize(posquare,otypes=[float])
print(vecfunc(poly1))

#第三个例子，比较三种+1方法
from time import time
def increByOne(x):
    return x+1
dataArray=np.linspace(1,5,1000000)
lenData=len(dataArray)
vecInc=np.vectorize(increByOne)
t1=time()
for i in range(lenData):
    dataArray[i]+=1
print('time for loop : %f'%(time()-t1))
t2=time()
dataArray=vecInc(dataArray)
print('time for vectorized ver-1: %f'%(time()-t2))
t3=time()
dataArray+=1
print('time for ndarray+1: %s'%(time()-t3))