import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
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



np.random.seed(1)
input_dim=2
num_classes=3
X,Y=generate(2000,input_dim,num_classes,[[3,3],[3,0]],False)
tmp=[np.argmax(i) for i in Y]
color_selects=['r','b','y']
colors=[color_selects[int(i)] for i in tmp]
plt.scatter(X[:,0],X[:,1],c=colors)
plt.show()

global_step=tf.Variable(0)
add_global=global_step.assign_add(1)

input_features=tf.placeholder(tf.float32,[None,input_dim])
input_labels=tf.placeholder(tf.float32,[None,num_classes])
W=tf.Variable(tf.random_normal([input_dim,num_classes]),name="weight")
b=tf.Variable(tf.zeros([num_classes]),name="bias")
z=tf.matmul(input_features,W)+b
pred=tf.nn.softmax(z)
err=tf.count_nonzero(tf.argmax(pred,axis=1)-tf.argmax(input_labels,axis=1))
cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=input_labels,logits=z)
loss=tf.reduce_mean(cross_entropy)
init_learning_rate=0.04
learning_rate=tf.train.exponential_decay(init_learning_rate,global_step,decay_steps=50,decay_rate=0.9)
optimizer=tf.train.AdamOptimizer(learning_rate)
train=optimizer.minimize(loss)

maxEpochs=50
miniBatchSize=25
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(maxEpochs):
        sess.run(add_global)
        sum_err=0
        sum_loss=0
        cnt=np.int32(np.ceil(len(Y)/miniBatchSize))
        for i in range(cnt):
            x1=X[i*miniBatchSize:(i+1)*miniBatchSize]
            y1=Y[i*miniBatchSize:(i+1)*miniBatchSize]

            _,lossval,outputval,errval=sess.run([train,loss,z,err],feed_dict={input_features:x1,input_labels:y1})
            sum_err+=errval
            sum_loss+=lossval
        sum_err/=len(Y)
        sum_loss/=len(Y)
        print("Epoch:","%04d"%(epoch+1),"cost=","{:.6f}".format(sum_loss),"err=","{:.6f}".format(sum_err))

    rst_W = sess.run(W)
    rst_b = sess.run(b)
    print("Fnish!")
    print("rst_W=", rst_W)
    print("rst_b=", rst_b)
    print("Accuracy on train:","{:.6f}".format(1 - sess.run(err, feed_dict={input_features: X, input_labels: Y}) / len(Y)))
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    x = np.linspace(-2, 8, 200)
    y1 = -x * (rst_W[0][0] / rst_W[1][0]) - rst_b[0] / rst_W[1][0]
    y2 = -x * (rst_W[0][1] / rst_W[1][1]) - rst_b[1] / rst_W[1][1]
    y3 = -x * (rst_W[0][2] / rst_W[1][2]) - rst_b[2] / rst_W[1][2]
    plt.plot(x, y1, label="first line", lw=3)
    plt.plot(x, y2, label="second line", lw=2)
    plt.plot(x, y3, label="third line", lw=1)
    plt.legend()
    plt.title("result on train data")
    plt.show()

    test_size=200
    test_X,test_Y=generate(test_size,input_dim,num_classes,[[3,3],[3,0]],False)
    print("Accuracy on test:","{:.6f}".format(1-sess.run(err,feed_dict={input_features:test_X,input_labels:test_Y})/test_size))
    tmp = [np.argmax(i) for i in test_Y]
    colors = [color_selects[int(i)] for i in tmp]
    plt.figure()
    plt.scatter(test_X[:,0],test_X[:,1],c=colors)
    plt.plot(x,y1,label="first line",lw=3)
    plt.plot(x, y2, label="second line", lw=2)
    plt.plot(x, y3, label="third line", lw=1)
    plt.legend()
    plt.title("result on test data")
    plt.show()

    plt.figure()
    plt.scatter(test_X[:, 0], test_X[:, 1], c=colors)
    nb_of_xs=20
    nb_of_ys=20
    xs1=np.linspace(-2,8,nb_of_xs)
    xs2=np.linspace(-2,8,nb_of_ys)
    xx,yy=np.meshgrid(xs1,xs2)#创建网格
    #填充
    classification_phone=np.zeros((nb_of_xs,nb_of_ys))
    for i in range(nb_of_xs):
        print(i)
        for j in range(nb_of_ys):
            classification_phone[i,j]=sess.run(tf.argmax(pred,axis=1),feed_dict={input_features:[[xx[i,j],yy[i,j]]]})
    #创建color map
    from matplotlib.colors import colorConverter, ListedColormap
    cmap=ListedColormap([colorConverter.to_rgba('r',alpha=0.3),colorConverter.to_rgba('b',alpha=0.3),colorConverter.to_rgba('y',alpha=0.3)])
    plt.contourf(xx,yy,classification_phone,cmap=cmap)#显示边界
    plt.show()







