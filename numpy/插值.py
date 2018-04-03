import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,8*np.pi,100)#从0~8π线性分割为100份
y=np.sin(x/2)
yinterp=np.interp(x,x,y)#一维分段线性插值得到新的y值
plt.figure(1)
plt.plot(x,y,'o')
plt.plot(x,yinterp,'x')
plt.show()
