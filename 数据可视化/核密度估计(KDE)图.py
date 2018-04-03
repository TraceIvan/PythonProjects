from numpy.random import  randn
import matplotlib as mpl
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats,integrate

sns.set(style="white", palette="muted", color_codes=True)
mpl.rc("figure",figsize=(10,6))
data=randn(250)
plt.title("KDE Demonstration using Seaborn and Matplotlib",fontsize=20)
sns.distplot(data,color='#ff8000',kde=True,rug=True,fit=stats.gamma)
print(help(sns.kdeplot))
plt.grid()
plt.show()

#绘制基本的单变量密度
plt.figure(2)
np.random.seed(10)
sns.set(color_codes=True)
mean, cov = [0, 2], [(1, .5), (.5, 1)]
x, y = np.random.multivariate_normal(mean, cov, size=50).T
ax = sns.kdeplot(x)
plt.show()
#绘制阴影，且使用不同的颜色
plt.figure(3)
ax = sns.kdeplot(x, shade=True, color="r")
plt.show()
#绘制双变量密度
plt.figure(4)
ax = sns.kdeplot(x, y)
plt.show()
#使用填充轮廓
plt.figure(5)
ax = sns.kdeplot(x, y, shade=True)
plt.show()
#使用更多的轮廓线和不同的调色板:
plt.figure(6)
ax = sns.kdeplot(x, y, n_levels=30, cmap="Purples_d")
plt.show()
#使用窄bandwith
plt.figure(7)
ax = sns.kdeplot(x, bw=.15)
plt.show()
#在垂直轴上绘制密度
plt.figure(8)
ax = sns.kdeplot(y, vertical=True)
plt.show()
#在数据范围内限制密度曲线:
plt.figure(9)
ax = sns.kdeplot(x, cut=0)
plt.show()
#为轮廓添加一个colorbar:
plt.figure(10)
ax = sns.kdeplot(x, y, cbar=True)
plt.show()
#绘制2个带有阴影的双变量密度
plt.figure(11)
iris = sns.load_dataset("iris")
setosa = iris.loc[iris.species == "setosa"]
virginica = iris.loc[iris.species == "virginica"]
ax = sns.kdeplot(setosa.sepal_width, setosa.sepal_length,
                 cmap="Reds", shade=True, shade_lowest=False)
ax = sns.kdeplot(virginica.sepal_width, virginica.sepal_length,
                 cmap="Blues", shade=True, shade_lowest=False)
plt.show()