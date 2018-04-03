import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#与学生花费在计算机上的时间相比，沿x轴的gpa的散点图
students=pd.read_csv('ucdavis.csv')
g=sns.FacetGrid(students,hue='gender',palette='Set1',size=6)#对数据集中一个变量的分布或多个变量间的关系进行可视化
g.map(plt.scatter,"momheight","height",s=250,linewidth=0.65,edgecolor='#ffad40')
g.set_axis_labels("Mothers Height","Students Height")
g.add_legend()
plt.show()
