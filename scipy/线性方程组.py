'''
求解：
x+2y-z=2
2x-3y+2z=2
3x+y-z=2
'''
import numpy as np
#系数矩阵
A=np.array([[1,2,-1],
           [2,-3,2],
           [3,1,-1]])
#结果矩阵
b=np.array([2.,2.,2.])
print(type(b))
v=np.linalg.solve(A,b)
print('the solution is ',v)
r1=np.dot(A,v)
