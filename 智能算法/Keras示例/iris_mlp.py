from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.models import load_model
import matplotlib.pyplot as plt
import csv
import numpy as np
# 保存loss和acc，画图
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # train acc 训练集准确率 accuracy
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val acc 验证集准确率 validation
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper left")
        plt.show()


# 获取数据
data = load_iris()
import numpy as np



# print(data)
# print(type(data))
x = data['data']
# print(x[1])
y = data['target']
# 训练集测试集划分 | random_state：随机数种子
x_train, x_test, y_init_train, y_init_test = train_test_split(x, y, test_size=0.2, random_state=1)
# 查看第一个样本
print(x_test[:1])
print(y_init_test[:1])
print(x_train.shape)
#MLP多分类
#x_train = np.random.random((1000, 20))
#y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
#x_test = np.random.random((100, 20))
#y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
#MLP二分类
#x_train = np.random.random((1000, 20))
#y_train = np.random.randint(2, size=(1000, 1))
#x_test = np.random.random((100, 20))
#y_test = np.random.randint(2, size=(100, 1))


# one hot编码
y_train = keras.utils.to_categorical(y_init_train, num_classes=3)
print(y_train.shape)
y_test = keras.utils.to_categorical(y_init_test, num_classes=3)
print(y_test[:1])

'''
基于MLP（Multi-layer Perceptron多层前馈神经网络）多层感知器的softmax多分类
'''
model = Sequential()
# Dense(128) is a fully-connected layer with 128 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 4-dimensional vectors.
#input_dim通常取特征维数
model.add(Dense(128, activation='relu', input_dim=4))
# Dropout随机失活，常用于图像识别中，防止过拟合
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

# lr表示学习速率，momentum表示动量项，decay是学习速率的衰减系数(每个epoch衰减一次)
# Nesterov的值是False或者True，表示使不使用Nesterov momentum
# SGD随机梯度下降
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# 编译模型
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# 创建一个实例history
history = LossHistory()

# 训练
model.fit(x_train, y_train,
          epochs=30,  # 数据被轮30次
          batch_size=128,
          validation_data=(x_test, y_test),
          callbacks=[history])
# 保存模型
# model.save('iris.h5')
# 读取模型
# model = load_model('iris.h5')

score = model.evaluate(x_test, y_test, verbose=0, batch_size=128)  # 不写默认是verbose=1，打印进度条记录，0不打印。
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# p_pred = model.predict(x_test)
# print("p_pred:\n", p_pred)
label_pred = model.predict_classes(x_test, verbose=0)
print("label_pred4test:\n", label_pred)
print("label_init4test:\n", y_init_test)
label_pred4train = model.predict_classes(x_train, verbose=0)
print("label_pred4train:\n", label_pred4train)
print("label_init4train:\n", y_init_train)

# 绘制acc-loss曲线
history.loss_plot('epoch')