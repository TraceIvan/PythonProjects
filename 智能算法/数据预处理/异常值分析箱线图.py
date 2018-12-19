import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']#正常显示中文标签
plt.rcParams['axes.unicode_minus']=False#正常显示正负号

path='D:\编程资料\机器学习\数据挖掘\《Python数据分析与挖掘实战》\图书配套数据、代码\chapter3\demo\data\catering_sale.xls'
data=pd.read_excel(path,index_col='日期')#指定日期为索引列
statistics=data.describe()
statistics.loc['range'] = statistics.loc['max']-statistics.loc['min'] #极差
statistics.loc['var'] = statistics.loc['std']/statistics.loc['mean'] #变异系数
statistics.loc['dis'] = statistics.loc['75%']-statistics.loc['25%'] #四分位数间距
print(statistics) #count为非空值的个数

plt.figure()
p=data.boxplot(return_type = 'dict')#绘制箱线图
x=p['fliers'][0].get_xdata()#异常值x坐标
y=p['fliers'][0].get_ydata()#异常值y坐标
plt.show()

print(data['销量'].values.tolist())

