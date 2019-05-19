import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from matplotlib.colors import colorConverter, ListedColormap

# 对于上面的fit可以这么扩展变成动态的
from sklearn.preprocessing import OneHotEncoder
def onehot(y, start, end):
    ohe = OneHotEncoder()
    a = np.linspace(start, end - 1, end - start)
    b = np.reshape(a, [-1, 1]).astype(np.int32)
    ohe.fit(b)
    c = ohe.transform(y).toarray()
    return c

def generate(sample_size, dims,num_classes, diff, regression=False):
    np.random.seed(10)
    mean = np.random.randn(dims)
    cov = np.eye(dims)
    # len(diff)
    samples_per_class = int(sample_size / num_classes)
    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)
    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean + d, cov, samples_per_class)
        Y1 = (ci + 1) * np.ones(samples_per_class)
        X0 = np.concatenate((X0, X1))
        Y0 = np.concatenate((Y0, Y1))
        # print(X0, Y0)
    if regression == False:  # one-hot  0 into the vector "1 0
        Y0 = np.reshape(Y0, [-1, 1])
        # print(Y0.astype(np.int32))
        Y0 = onehot(Y0.astype(np.int32), 0, num_classes)
        # print(Y0)
    X, Y = shuffle(X0, Y0)
    # print(X, Y)
    return X, Y

np.random.seed(10)
X,Y=generate(1000,2,2,[[3,3]],True)
colors=['r' if l==0 else 'b' for l in Y[:]]
plt.scatter(X[:,0],X[:,1],c=colors)
plt.xlabel("Scaled age (in yrs)")
plt.ylabel("Tumor size(in cm)")
plt.show()

#输入结点
input_dim = 2
lab_dim=1
input_features=tf.placeholder(tf.float32,[None,input_dim])
input_lables=tf.placeholder(tf.float32,[None,lab_dim])
#参数
W=tf.Variable(tf.random_normal([input_dim,lab_dim]),name="weight")
b=tf.Variable(tf.zeros([lab_dim]),name="bias")
#输出结点
output=tf.nn.sigmoid(tf.matmul(input_features,W)+b)
#损失
cross_entropy=-(input_lables*tf.log(output)+(1-input_lables)*tf.log(1-output))
ser=tf.square(input_lables-output)
loss=tf.reduce_mean(cross_entropy)
err=tf.reduce_mean(ser)
optimizer=tf.train.AdamOptimizer(0.04)
train=optimizer.minimize(loss)

maxEpochs=50
minibatchSize=25
#启动Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(maxEpochs):
        sum_err=0
        for i in range(np.int32(len(Y)/minibatchSize)):
            x1=X[i*minibatchSize:(i+1)*minibatchSize,:]
            y1=np.reshape(Y[i*minibatchSize:(i+1)*minibatchSize],[-1,1])
            tf.reshape(y1,[-1,1])
            _,lossval,outputval,errval=sess.run([train,loss,output,err],feed_dict={input_features:x1,input_lables:y1})
            sum_err+=errval
        print("Epoch:","%04d"%(epoch+1),"cost=","{:.9f}".format(lossval),"err=",sum_err/minibatchSize)

    #可视化
    plt.scatter(X[:,0],X[:,1],c=colors)
    x=np.linspace(-1,8,200)
    #    x1w1+x2*w2+b=0
    #    x2=-x1* w1/w2-b/w2
    y=-x*(sess.run(W)[0]/sess.run(W)[1])-sess.run(b)/sess.run(W)[1]
    plt.plot(x,y,label="Fitted line")
    plt.legend()
    plt.show()
