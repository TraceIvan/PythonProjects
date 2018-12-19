import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
plt.rcParams['font.sans-serif']=['SimHei']#正常显示中文标签
plt.rcParams['axes.unicode_minus']=False#正常显示正负号

path='D:\编程资料\机器学习\数据挖掘\《Python数据分析与挖掘实战》\图书配套数据、代码\chapter3\demo\data\catering_sale.xls'
data=pd.read_excel(path,index_col='日期')#指定日期为索引列
data_2=data.dropna()#去除缺失值

sells=data_2['销量'].values.tolist()
statistics = data_2.describe() #保存基本统计量
MaxMin=float(statistics.loc['max']-statistics.loc['min'])#极差
Count=int(statistics.loc['count'])#总数
dis=1000#组距
cnt=int(ceil(MaxMin/dis))#组数
x=[]
y=[]
tmp=0
for i in range(cnt):
    x.append(str(tmp)+'~'+str(tmp+dis))
    tmp+=dis
    y.append({'nums':0,'mid':0,'tot':[],'f':0,'cul_f':0})
for i in range(Count):
    tmp=sells[i]
    pos=int(tmp//dis)
    y[pos]['nums']+=1
    y[pos]['tot'].append(tmp)
pre=0#累计频率
for i in range(cnt):
    if y[i]['nums']:
        y[i]['mid']=sum(y[i]['tot'])/y[i]['nums']
    y[i]['f']=y[i]['nums']/Count
    pre=y[i]['cul_f']=y[i]['f']+pre

for i in range(cnt):
    print(x[i])
    print(y[i])

y_nums=[]
for i in range(cnt):
    y_nums.append(y[i]['f'])
#绘制直方图
plt.figure()
plt.bar(range(cnt),y_nums)
plt.xticks(range(cnt),x)#设置x轴刻度显示
ax = plt.gca()
for label in ax.xaxis.get_ticklabels():
    label.set_rotation(30)
plt.xlabel('销量')
plt.ylabel('频率')
plt.title('销量频率分布直方图')
plt.show()