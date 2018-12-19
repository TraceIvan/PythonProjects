import pandas as pd
import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']#正常显示中文标签
plt.rcParams['axes.unicode_minus']=False#正常显示正负号

#计算n阶差商
def get_n_diff_quot(x,y,n,f):
    for i in range(n-1):
        for j in range(n-1-i):
            if i==0:
                f[i][j]=(y[j]-y[j+1])/(x[j]-x[j+1])
            else:
                f[i][j]=(f[i-1][j]-f[i-1][j+1])/(x[j]-x[j+1+i])

def get_Px(c_x,f,x,y0,n):
    ret=y0*1.0
    tmp=1.0
    for i in range(n-1):
        tmp=tmp*(c_x-x[i])
        t_ret=1.0*tmp*f[i][0]
        ret=1.0*ret+t_ret
    return ret


x=[i for i in range(-50, 51, 10)]
t_x=[i for i in range(-50, 51)]
y=[i**2 for i in x]

n=len(x)
f=[[0 for i in range(n)] for j in range(n)]
get_n_diff_quot(x,y,n,f)
t_y=[get_Px(i,f,x,y[0],n) for i in t_x]

plt.figure()
plt.plot(x, y, linestyle = '', marker='o', color='b')
plt.plot(t_x, t_y, linestyle = '--', color='r')
plt.show()
