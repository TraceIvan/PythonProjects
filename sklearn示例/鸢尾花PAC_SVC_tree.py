import matplotlib.pyplot as plt                 #加载matplotlib用于数据的可视化
from sklearn.decomposition import PCA           #加载PCA算法包
from sklearn.datasets import load_iris
import numpy as np
import matplotlib as mpl
from matplotlib import colors
import pylab as pl


data=load_iris()
iris_y=data.target
iris_x=data.data
print(iris_x)
print(iris_y)
#-----------------------鸢尾花降维————————
pca=PCA(n_components=2)     #加载PCA算法，设置降维后主成分数目为2
reduced_x=pca.fit_transform(iris_x)#对样本进行降维

red_x,red_y=[],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]


for i in range(len(reduced_x)):
    if iris_y[i] ==0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])

    elif iris_y[i]==1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])

    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])

#可视化
plt.scatter(red_x,red_y,c='r',marker='x')
plt.scatter(blue_x,blue_y,c='b',marker='D')
plt.scatter(green_x,green_y,c='g',marker='.')
plt.show()

#-----SVC-----
from sklearn import svm
from sklearn import model_selection


classifier=svm.SVC(kernel='rbf',gamma=0.1,decision_function_shape='ovr',C=0.1)
'''
kernel: 'rbf'--高斯核，'linear'--线性核
C越大分类效果越好，但有可能会过拟合（defaul C=1）
gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合
decision_function_shape：'ovo'--one v one分类问题，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
                         'ovr'--为one v rest分类问题，即一个类别与其他类别进行划分

'''
'''# 用train_test_split将数据随机分为训练集和测试集，测试集占总数据的30%（test_size=0.3),random_state是随机数种子'''
x_train,x_test,y_train,y_test=model_selection.train_test_split(iris_x,iris_y,random_state=1,test_size=0.3)
classifier.fit(x_train,y_train.ravel())
'''
.flatten() 返回一份拷贝，对拷贝所做的修改不会影响（reflects）原始矩阵
.ravel()返回的是视图（view），会影响（reflects）原始矩阵
'''
print('训练集 decision_function:\n', classifier.decision_function(x_train))
print('\n训练集 predict:\n', classifier.predict(x_train))
print("SVM-输出训练集的准确率为：",classifier.score(x_train,y_train))
print("SVM-输出测试集的准确率为：",classifier.score(x_test,y_test))

x_train,x_test,y_train,y_test=model_selection.train_test_split(reduced_x,iris_y,random_state=1,test_size=0.3)
classifier.fit(x_train,y_train.ravel())
print("降维后，SVM-输出训练集的准确率为：",classifier.score(x_train,y_train))
print("降维后，SVM-输出测试集的准确率为：",classifier.score(x_test,y_test))
n_Support_vector = classifier.n_support_#支持向量个数
print("支持向量个数为： ",n_Support_vector)
Support_vector_index = classifier.support_#支持向量索引
#绘制降维后SVM图形
N, M = 1000, 1000  # 横纵各采样多少个值
x1_min, x2_min = reduced_x.min(axis=0)-1
x1_max, x2_max = reduced_x.max(axis=0)+1
t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, M)
x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
y_predict=classifier.predict(x_show)
#指定默认字体
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
#绘制
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
alpha=0.5
plt.pcolormesh(x1, x2,y_predict.reshape(x1.shape), cmap=cm_light) # 预测值的显示
plt.scatter(reduced_x[:,0],reduced_x[:,1],c=iris_y,cmap=cm_dark,marker='o',edgecolors='k')
for j in Support_vector_index:
    plt.scatter(reduced_x[j,0],reduced_x[j,1], s=100, c = '', alpha=0.5, linewidth=1.5, edgecolor='white')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title("降维后SVM结果示意")
plt.show()
#------------决策树-----------
from sklearn import tree
clf=tree.DecisionTreeClassifier(max_depth=4)#基于CART树,采用基尼系数
'''
criterion:"gini"或者"entropy"，前者代表基尼系数，后者代表信息增益
splitter:"best"或者"random"。前者在特征的所有划分点中找出最优的划分点。后者是随机的在部分划分点中找局部最优的划分点
'''
clf.fit(iris_x,iris_y)
print(clf.score(iris_x,iris_y))
print(clf.predict([[7,7,7,7]]))
# 画图(根据降维数据)
clf2=tree.DecisionTreeClassifier(max_depth=3)
clf2.fit(reduced_x,iris_y)
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                     np.arange(x2_min, x2_max, 0.1))
xxs=np.stack((xx.flat, yy.flat), axis=1)
Z = clf2.predict(xxs)
plt.contourf(xx,yy, Z.reshape(xx.shape), alpha=0.4)
plt.scatter(reduced_x[:, 0], reduced_x[:, 1], c=iris_y, alpha=0.8)
plt.show()
# 导出 .dot文件
with open("tree.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
#dot -Tpdf tree.dot -o tree.pdf
feature_name =['花萼长度','花萼宽度','花瓣长度','花瓣宽度']
#class_name=['0_setosa','1_versicolor','2_virginica']
class_name=["山鸢尾","虹膜锦葵","变色鸢尾"]
from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,feature_names=feature_name,class_names=class_name,filled=True, rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue().replace('helvetica',' "Microsoft YaHei" '))
graph.write_pdf("iris.pdf")

#--------KNN------------
from sklearn.neighbors import KNeighborsClassifier
x_train,x_test,y_train,y_test=model_selection.train_test_split(iris_x,iris_y,random_state=1,test_size=0.3)
knn = KNeighborsClassifier(n_neighbors=5) # 创建一个KNN算法实例
knn.fit(x_train, y_train)
y_predict = knn.predict(x_test) # 获取预测结果
for i in range(len(y_predict)):
    print("第%d次测试:真实值:%s\t预测值:%s"%((i+1),class_name[y_predict[i]],class_name[y_test[i]]))
print(knn.score(x_test,y_test))

#------logistic回归-----
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x_train)
x_train_std=sc.transform(x_train)#标准化
x_test_std=sc.transform(x_test)
lr=LogisticRegression(C=100.0,random_state=0)
lr.fit(x_train_std,y_train)
y_pred=lr.predict(x_test_std)
print (y_pred)
print(lr.score(x_test_std,y_test))