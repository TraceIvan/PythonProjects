from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np  # 转换成数组
import matplotlib.pyplot as plt

#线性回归
iris = load_iris() # 获取花瓣的长和宽
x = [n[0] for n in iris.data]
y = [n[1] for n in iris.data]

x = np.array(x).reshape(len(x), 1)
y = np.array(y).reshape(len(y), 1)

clf = LinearRegression()
clf.fit(x, y)
pre = clf.predict(x)

plt.scatter(x, y, s=100)
plt.plot(x, pre, "r-", linewidth=4)
for idx, m in enumerate(x):
    plt.plot([m, m], [y[idx], pre[idx]], 'g-')
plt.show()

#决策树
clf = DecisionTreeClassifier()
clf.fit(iris.data, iris.target)
print(clf)
predicted = clf.predict(iris.data)

#获取花卉两列数据集  
X = iris.data
L1 = [x[0] for x in X]
print(L1)
L2 = [x[1] for x in X]
print(L2)
plt.scatter(L1, L2, c=predicted, marker='x')
plt.title("DTC")
plt.show()
