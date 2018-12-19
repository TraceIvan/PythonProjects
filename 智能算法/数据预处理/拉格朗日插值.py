import pandas as pd
import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']#正常显示中文标签
plt.rcParams['axes.unicode_minus']=False#正常显示正负号

path='D:\编程资料\机器学习\数据挖掘\《Python数据分析与挖掘实战》\图书配套数据、代码\chapter3\demo\data\catering_sale.xls'
outputpath='sales.xls'

data=pd.read_excel(path)
#插值前通常会对数据进行异常值检测，将其定义为空缺值
#data['销量'].apply(lambda x:None if x<400 or x>5000 else x)#过滤异常值，将其变为空值
print(data['销量'])
data['销量']=data.loc[:,'销量'].apply(lambda x:None if x<400 or x>5000 else x)
print(data['销量'])

#自定义列向量插值函数
#s为列向量，n为被插值的位置，k为取前后的数据个数，默认为5
def ployinterp_column(s,n,k=5):
    y=s[list(range(max(n-k,0),n))+list(range(n+1,min(n+1+k,len(s))))]#取值
    y=y[y.notnull()]#剔除空值
    poly=lagrange(y.index,list(y))
    return poly(n)#插值并返回结果

#逐个元素判断是否需要差值
# for i in data.columns:
#     for j in range(len(data)):
#         if (data[i].isnull())[j]:#如果为空则插值
#             data.loc[i,j]=ployinterp_column(data[i],j)
for i in range(len(data)):
    if np.isnan(data.loc[i,'销量']):
        data.loc[i, '销量'] = ployinterp_column(data['销量'], i)
data.to_excel(outputpath)


