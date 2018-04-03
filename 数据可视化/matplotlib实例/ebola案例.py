import csv
import pandas as pd
import operator
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
#用csv读取csv文件
'''该csv包含指标、国家名称、日期和与指标相关的死亡人数或感染数量'''
with open('ebola_data_db_format.csv','rt') as f:
    filtereddata=[row for row in csv.reader(f) if row[3]!="0.0" and row[3]!="0"
                  and "deaths" in row[0]]#读取死亡人数
#用pandas读取csv文件
eboladata=pd.read_csv('ebola_data_db_format.csv')
filtered=eboladata[eboladata["value"]>0]
filtered=filtered[filtered["Indicator"].str.contains("deaths")]
#按国家列分类
sorteddata=sorted(filtereddata,key=operator.itemgetter(1))
#选出几内亚、利比里亚以及塞拉利昂这三个有较多死亡人数的国家
guineadata=[row for row in sorteddata if row[1]=='Guinea' and row[0]=="Cumulative number" \
                                                                      " of confirmed Ebola" \
                                                                      " deaths"]
guineadata=sorted(guineadata,key=operator.itemgetter(2))
sierradata=[row for row in sorteddata if row[1]=='Sierra Leone' and row[0]=="Cumulative number" \
                                                                      " of confirmed Ebola" \
                                                                      " deaths"]
sierradata=sorted(sierradata,key=operator.itemgetter(2))
liberiadata=[row for row in sorteddata if row[1]=='Liberia' and row[0]=="Cumulative number" \
                                                                      " of confirmed Ebola" \
                                                                      " deaths"]
liberiadata=sorted(liberiadata,key=operator.itemgetter(2))
#获取三个国家对应日期的死亡人数
g_x=[dt.datetime.strptime(row[2],'%Y-%m-%d').date() for row in guineadata]
g_y=[float(row[3]) for row in guineadata]
s_x=[dt.datetime.strptime(row[2],'%Y-%m-%d').date() for row in sierradata]
s_y=[float(row[3]) for row in sierradata]
l_x=[dt.datetime.strptime(row[2],'%Y-%m-%d').date() for row in liberiadata]
l_y=[float(row[3]) for row in liberiadata]
#画图
plt.figure(1)
plt.plot(g_x,g_y,color='red',linewidth=2,label='Guinea')
plt.plot(s_x,s_y,color='orange',linewidth=2,label='Sierra Leone')
plt.plot(l_x,l_y,color='blue',linewidth=2,label='Liberia')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Numbers of Ebola Deaths',fontsize=18)
plt.title('Confirmed Ebola Deaths',fontsize=20)
plt.legend(loc=2)
plt.show()
#绘制指标为"Cumulative number of confirmed, probable and suspected Ebola deaths"的折线图
guineadata2=[row for row in sorteddata if row[1]=='Guinea' and row[0]=="Cumulative number of confirmed, " \
                                                                      "probable and suspected Ebola deaths"]
guineadata2=sorted(guineadata2,key=operator.itemgetter(2))
sierradata2=[row for row in sorteddata if row[1]=='Sierra Leone' and row[0]=="Cumulative number of confirmed, " \
                                                                            "probable and suspected Ebola deaths"]
sierradata2=sorted(sierradata2,key=operator.itemgetter(2))
liberiadata2=[row for row in sorteddata if row[1]=='Liberia' and row[0]=="Cumulative number of confirmed, " \
                                                                            "probable and suspected Ebola deaths"]
liberiadata2=sorted(liberiadata2,key=operator.itemgetter(2))
#获取三个国家对应日期的死亡人数
g_x2=[dt.datetime.strptime(row[2],'%Y-%m-%d').date() for row in guineadata2]
g_y2=[float(row[3]) for row in guineadata2]
s_x2=[dt.datetime.strptime(row[2],'%Y-%m-%d').date() for row in sierradata2]
s_y2=[float(row[3]) for row in sierradata2]
l_x2=[dt.datetime.strptime(row[2],'%Y-%m-%d').date() for row in liberiadata2]
l_y2=[float(row[3]) for row in liberiadata2]
#画图
plt.figure(2)
plt.plot(g_x2,g_y2,color='red',linewidth=2,label='Guinea')
plt.plot(s_x2,s_y2,color='orange',linewidth=2,label='Sierra Leone')
plt.plot(l_x2,l_y2,color='blue',linewidth=2,label='Liberia')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Numbers of Ebola Deaths',fontsize=18)
plt.title('Probable and Suspected Ebola Deaths',fontsize=20)
plt.legend(loc=2)
plt.show()
