import numpy as np
import matplotlib.pyplot as plt
def get_data():
    np.random.seed(37)  # 使得每次运行得到的随机数都一样
    x = np.arange(10, 20)  # 自变量，随便定义的
    error = np.random.normal(size=x.shape)
    y = 1.8 * x + 5.9 + error  # 添加随机数作为噪音
    plt.figure(1)
    plt.scatter(x, y)
    plt.plot(x, 1.8 * x + 5.9, '-r')  # 绘制的是红色的直线
    plt.show()
    # 以下加入两个异常点
    abnormal_x = [16.5, 17.9]
    abnormal_y = [25.98, 24.12]
    # 将异常点绘制出来
    plt.figure(2)
    plt.scatter(abnormal_x, abnormal_y, marker='x')
    # 将异常点加入到原数据集中，构建线性回归器进行拟合，绘制拟合直线
    whole_x = np.append(x, abnormal_x).reshape(-1, 1)
    whole_y = np.append(y, abnormal_y).reshape(-1, 1)
    #构建测试数据
    test_x = np.arange(10, 20)  # 自变量，随便定义的
    shift = np.random.normal(size=test_x.shape)
    test_x = test_x + shift  # 对test_x进行偏置得到测试集的X
    error = np.random.normal(size=x.shape)
    test_y = 1.8 * test_x + 5.9 + error  # 添加随机数作为噪音
    plt.scatter(whole_x, whole_y, color='blue', label='train_set')
    plt.scatter(test_x, test_y, color='red', label='test_set')
    plt.legend()
    plt.show()
    # 把train set和test set都绘制到一个图中，可以看出偏差不大
    return whole_x,whole_y,test_x,test_y

def Ridge_Regressor(whole_x,whole_y):
    # 岭回归器的构建
    #alpha参数控制岭回归器的复杂程度，但alpha趋近于0时，岭回归器就相当于普通的最小二乘法
    from sklearn import linear_model
    ridge_regressor = linear_model.Ridge(alpha=0.02, fit_intercept=True, max_iter=10000)
    # 构建岭回归器对象，使用的偏差系数为alpha=0.02
    ridge_regressor.fit(whole_x, whole_y)  # 使用岭回归器进行训练

    # 使用训练完成的岭回归器预测数值
    y_train_predict = ridge_regressor.predict(whole_x)
    plt.scatter(whole_x, whole_y)
    plt.plot(whole_x, y_train_predict, '-r')
    plt.show()
    return ridge_regressor

def Assess(test_x,test_y,ridge_regressor):
    # 岭回归器模型的评估
    y_test_predict = ridge_regressor.predict(test_x.reshape(-1, 1))
    # 使用评价指标来评估模型的好坏
    import sklearn.metrics as metrics
    test_y = test_y.reshape(-1, 1)
    print('平均绝对误差：{}'.format(
        round(metrics.mean_absolute_error(y_test_predict, test_y), 2)))
    print('均方误差MSE：{}'.format(
        round(metrics.mean_squared_error(y_test_predict, test_y), 2)))
    print('中位数绝对误差：{}'.format(
        round(metrics.median_absolute_error(y_test_predict, test_y), 2)))
    print('解释方差分：{}'.format(
        round(metrics.explained_variance_score(y_test_predict, test_y), 2)))
    print('R方得分：{}'.format(
        round(metrics.r2_score(y_test_predict, test_y), 2)))

if __name__=='__main__':
    whole_x,whole_y,test_x,test_y=get_data()
    print(whole_x)
    print(whole_y)
    print(test_x)
    print(test_y)
    ridge_regressor=Ridge_Regressor(whole_x,whole_y)
    Assess(test_x,test_y,ridge_regressor)

