import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from matplotlib.ticker import LinearLocator,FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
#产生X和Y网格
X=np.arange(-4,4,0.25)
Y=np.arange(-4,4,0.25)
X,Y=np.meshgrid(X,Y)
#产生Z轴数据
R=np.sqrt(X**2,Y**2)
Z=np.sin(R)
#绘制三维图
fig=plt.figure()
ax=fig.gca(projection='3d')
surf=ax.plot_surface(X,Y,Z,rstride=1,cstride=1,linewidth=0,cmap=cm.coolwarm,antialiased=False)
ax.set_zlim(-1.01,1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf,shrink=0.6,aspect=6)
plt.show()
