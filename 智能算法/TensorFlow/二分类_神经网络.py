import tensorflow as tf
from numpy.random import RandomState#通过numpy生成模拟数据集

#定义训练数据batch大小
batch_size=8

#定义神经网络的参数
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))#seed设置使得每次运行该网络结果相同
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#在shape的一个维度使用None可以方便使用不大的batch大小，在训练时需要
# 把数据分为比较小的batch，但在测试时，可以一次性使用全部的数据，当数据比较小
#时方便测试，但是数据集比较大时，将大量数据放入一个batch可能导致内存溢出
x=tf.placeholder(tf.float32,shape=(None,2),name='x-input')#提供输入数据
y_=tf.placeholder(tf.float32,shape=(None,1),name='y-input')

#定义神经网络前向传播的过程
a=tf.matmul(x,w1)#矩阵乘法
y=tf.matmul(a,w2)

#定义损失函数和反向传播的算法
cross_entropy= -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y,1e-10,1.0))
)#定义真实值和预测值之间的交叉熵,y_为正确结果，y为预测结果。tf.clip_by_value()将一个
#张量中的数值限制在一个范围内，保证不会出现log(0)或大于1的概率。tf.log取对数功能。
#得到n×m矩阵，n为一个batch中的样例数目，m为分类的类别数量
train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#通过随机数生成一个模拟数据集
rdm=RandomState(1)
dataset_size=128
X =rdm.rand(dataset_size,2)
#定义规则来给出样本的标签，此处所有x1+x2<1的样例被认为是正样本（比如零件合格）
#而其他样本为负样本（比如零件不合格）。这里用0表示负样本，用1表示正样本
Y =[[int(x1+x2<1)] for (x1,x2) in X]

#创建一个会话运行程序
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()#初始化变量
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))
    '''
    在训练之前神经网络参数的值：
    w1=[[-0.8113182   1.4845988   0.06532937]
        [-2.4427042   0.0992484   0.5912243 ]]
    w2=[[-0.8113182 ]
        [ 1.4845988 ]
        [ 0.06532937]]
    '''

    #设定训练的轮数
    STEPS=5000
    for i in range(STEPS+1):
        #每次选取batch_size个样本进行训练
        start=(i*batch_size)%dataset_size
        end=min(start+batch_size,dataset_size)

        #通过选取的样本训练神经网络并更新参数
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i %1000==0:
            #每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy=sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print("After %d training step(s), cross entropy on all data is %g"%
                  (i,total_cross_entropy))
            '''
            输出结果：
            After 0 training step(s), cross entropy on all data is 0.0674925
            After 1000 training step(s), cross entropy on all data is 0.0163385
            After 2000 training step(s), cross entropy on all data is 0.00907547
            After 3000 training step(s), cross entropy on all data is 0.00714436
            After 4000 training step(s), cross entropy on all data is 0.00578471
            After 5000 training step(s), cross entropy on all data is 0.00430222
            '''
    print(sess.run(w1))
    print(sess.run(w2))
    '''
    在训练之后神经网络参数的值：
    w1=[[-1.9621537  2.5826619  1.6824381]
        [-3.4685102  1.0701413  2.118307 ]]
    w2=[[-1.8250272]
        [ 2.6858096]
        [ 1.4185809]]
    '''
