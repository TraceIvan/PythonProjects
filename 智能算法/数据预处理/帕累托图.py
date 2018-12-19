import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
plt.rcParams['font.sans-serif']=['SimHei']#正常显示中文标签
plt.rcParams['axes.unicode_minus']=False#正常显示正负号

path='D:\编程资料\机器学习\数据挖掘\《Python数据分析与挖掘实战》\图书配套数据、代码\chapter3\demo\data\catering_dish_profit.xls'

data=pd.read_excel(path,index_col='菜品名')
profit=data['盈利'].copy()
profit.sort_values(ascending=False)#递减

plt.figure()
profit.plot(kind='bar')
plt.ylabel('盈利(元)')
p=1.0*profit.cumsum()/profit.sum()
p.plot(color='r',secondary_y=True,style='-o',linewidth=2)
plt.annotate(format(p[6],'.4%'),xy=(6,p[6]),xytext=(6*0.9,p[6]*0.9),arrowprops=dict(arrowstyle='->',connectionstyle="arc3,rad=.2"))#添加注释
plt.ylabel('盈利(比例)')
plt.title("菜品盈利帕累托图")
plt.show()