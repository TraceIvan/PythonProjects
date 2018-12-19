import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
plt.rcParams['font.sans-serif']=['SimHei']#正常显示中文标签
plt.rcParams['axes.unicode_minus']=False#正常显示正负号

x=pd.Series(np.exp(np.arange(20)))
x.plot(label="原始数据图",legend=True)
plt.show()
x.plot(logy=True,label='对数数据图',legend=True)
plt.show()

error=np.random.randn(10)#定义误差列
y=pd.Series(np.sin(np.arange(10)))#均值数据列
y.plot(yerr=error,fmt='.k',color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)
plt.show()
plt.errorbar(np.arange(10),y,yerr=error)
plt.show()