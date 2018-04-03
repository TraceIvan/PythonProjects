import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
fig,ax=plt.subplots()
ax.plot(np.random.rand(20),'-o',ms=20,lw=2,alpha=0.7,mfc='orange')
#置于底部右端
fig.text(0.95,0.05,'Property of NPL',fontsize=40,color='gray',
         ha='right',va='bottom',alpha=0.5)
plt.show()