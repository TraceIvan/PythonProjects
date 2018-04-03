import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-np.pi/2,np.pi/2,44)
y=1/(1+np.cos(x)*np.cos(x))#原始方程
dy_actual=np.sin(2*x)/(1+np.cos(x)*np.cos(x))**2#导数

fig=plt.figure(1)
ax=fig.add_subplot(111)
dy=np.zeros(y.shape,np.float)
dy[0:-1]=np.diff(y)/np.diff(x)#前向差分
dy[-1]=(y[-1]-y[-2])/(x[-1]-x[-2])

plt.plot(x,y,linewidth=3,color='b',label='actual function')
plt.plot(x,dy_actual,label='actual derivative',linewidth=2,color='r')
plt.plot(x,dy,label='forward diff',linewidth=2,color='g')
plt.legend(loc='upper center')
plt.show()
