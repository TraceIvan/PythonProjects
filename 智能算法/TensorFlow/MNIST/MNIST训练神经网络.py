'''
简介：三层全连接神经网络模型对MNIST数字进行识别
采用5种优化方法：
    神经网络结构方面：使用激活函数实现神经网络模型的去线性化
                      使用一个隐藏层使得结构更深，已解决复杂问题
    训练神经网络时：使用带指数衰减的学习率设置
                    使用正则化避免过度拟合
                    使用滑动平均模型使得最终模型更加健壮
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#MNIST数据集相关的常数
INPUT_NODE=784 #输入层的节点数。对于MNIST数据集，即为图片的像素
OUTPUT_NODE=10 #输出层的节点数。等于类别的数目，区分0~9这10个数字

#配置神经网络的参数
LAYER1_NODE=500 #隐藏层节点数。这里使用只有一个隐藏层的网络结构作为样例，有500个节点
BATCH_SIZE=100 #一个训练batch中的训练数据个数。数字越小时，训练过程越接近随机梯度下降；
               #数字越大时，训练越接近梯度下降
LEARNING_RATE_BASE=0.8 #基础的学习率
LEARNING_RATE_DECAY=0.99 #学习率的衰减率
REGULARIZATION_RATE=0.0001 #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS=30000 #训练轮数
MOVING_AVERAGE_DECAY=0.99 #滑动平均衰减率

'''
一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果。
激活函数选择：ReLU函数——去线性化
网络结构：三层全连接神经网络
支持传入用于计算参数平均值的类，方便测试时使用滑动平均模型
'''
def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    #当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class==None:
        #计算隐藏层前向传播结果，使用ReLU激活函数
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)

        #计算输出层的前向传播结果
        #这里不需加入激活函数，在计算损失函数时会一并运算softmax函数
        #而且不加入激活函数不会影响预测结果。预测时使用的是不同类别对应节点输出值的
        # 相对大小，有没有softmax层对最后分类结果的计算没有影响。于是计算整个神经网络
        # 的前向传播时可以不加入最后的softmax层
        return tf.matmul(layer1,weights2)+biases2
    else:
        #首先使用avg_class.average函数计算得出变量的滑动平均值，然后计算相应前向传播结果
        layer1=tf.nn.relu(
            tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1)
        )
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)

#训练模型的过程
def train(mnist):
    x=tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')

    #生成隐藏层参数
    weights1=tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))#stddev：标准差
    weights1=tf.get_variable("weights1",shape=[INPUT_NODE,LAYER1_NODE],initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))#偏置项
    #生成输出层参数
    weights2=tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    #计算在当前参数下神经网络前向传播的结果，这里给出的用于滑动平均的类为None,不会使用
    # 参数的滑动平均值
    y=inference(x,None,weights1,biases1,weights2,biases2)

    #定义存储训练轮数的变量
    #不需要计算滑动平均值，故指定为不可训练的变量
    #一般将代表训练轮数的变量指定为不可训练的参数
    global_step=tf.Variable(0,trainable=False)

    #给定滑动平均衰减率和训练轮数的变量
    #初始化滑动平均类
    #给定训练轮数的变量可以加快训练早期的更新速度
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

    #在所有代表神经网络参数的变量上使用滑动平均
    #辅助变量（如global_step）不需要
    #tf.trainable_variables返回的图上集合GraphKeys.TRAINABLE_VARIABLES中的元素
    #这个集合为所有没有指定trainable=False的参数
    variables_averages_op=variable_averages.apply(tf.trainable_variables())

    #计算使用了滑动平均之后的前向传播结果
    #滑动平均不会改变变量本身，而是维护一个影子变量记录滑动平均值
    #需要滑动平均值需要明确调用average函数
    average_y=inference(x,variable_averages,weights1,biases1,weights2,biases2)

    #计算交叉熵作为刻画预测值和真实值之间差距的损失函数
    #当分类问题只有一个正确答案时，可以采用sparse_softmax_cross_entropy_with_logits计算
    #参数：不包含softmax层的前向传播结果；训练数据的正确答案
    #标准答案为长度为10的一维数组，使用tf.argmax得到正确答案对应的类别编号
    '''
    tf.argmax(input, axis=None, name=None, dimension=None)
    此函数是对矩阵按行或列计算最大值
    参数:
    input：输入Tensor
    axis：0表示按列，1表示按行
    name：名称
    dimension：和axis功能一样，默认axis取值优先。新加的字段
    返回：Tensor  一般是行或列的最大值下标向量
    '''
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    #计算在当前batch中所有样例的交叉熵平均值
    '''
    tf.reduce_mean()
    如果不指定第二个参数，那么就在所有的元素中取平均值
    指定第二个参数为0，则第一维的元素取平均值，即每一列求平均值
    指定第二个参数为1，则第二维的元素取平均值，即每一行求平均值
    '''
    cross_entropy_mean=tf.reduce_mean(cross_entropy)

    #计算L2正则化损失函数
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算模型的正则化损失
    #一般只计算边权的正则化损失，而不使用偏置项
    regularization=regularizer(weights1)+regularizer(weights2)

    #总损失等于交叉熵损失和正则化损失的和
    loss=cross_entropy_mean+regularization

    #设置指数衰减的学习率
    learning_rate=tf.train.exponential_decay(
        LEARNING_RATE_BASE,#基础学习率，随着迭代的进行，更新变量时使用的学习率在这个基础上递减
        global_step,#当前迭代的轮数
        mnist.train.num_examples / BATCH_SIZE,#过完所有的训练数据需要的迭代次数
        LEARNING_RATE_DECAY#学习率衰减速度
    )

    #使用tf.train.GradientDescentOptimizer优化算法优化损失函数（包含交叉熵损失和L2正则化损失）
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    #只优化交叉熵:
    #train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean,global_step=global_step)

    #在训练神经网络模型时，每过一遍数据需要通过反向传播更新参数，又要更新参数的滑动平均值
    #为一次完成多次操作，采用tf.control_dependencies和tf.group两种机制
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op=tf.no_op(name='train')
    #上一句等价于train_op=tf.group(train_step,variables_averages_op)

    #检验使用滑动平均模型的神经网络模型前向传播结果是否正确
    correct_prediction=tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    '''
    tf.cast(x, dtype, name=None)
    此函数是类型转换函数
    参数
    x：输入
    dtype：转换目标类型
    name：名称
    返回：Tensor
    '''
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #初始化会话并开始训练过程：
    with tf.Session()  as sess:
        tf.global_variables_initializer().run()
        #准备验证数据
        #一般在神经网络训练过程中通过验证数据来大致判断停止的条件和评判训练的结果
        validate_feed={x:mnist.validation.images,y_:mnist.validation.labels}

        #准备测试数据
        test_feed={x:mnist.test.images,y_:mnist.test.labels}

        #迭代地训练神经网络
        for i in range(TRAINING_STEPS):
            #每1000轮输出一次在验证数据集上的测试数据
            if i%1000==0:
                #计算滑动平均模型在验证数据和测试数据上的结果
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                test_acc=sess.run(accuracy,feed_dict=test_feed)
                print("After %d training step(s), validation accuracy using average "
                      "model is %g, test accuracy using average model is %g."
                      %(i,validate_acc,test_acc))

            #产生这一轮使用的一个batch的训练数据，并运行训练过程
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})

        #训练结束后，在测试数据上检测神经网络模型的最终正确率
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average "
              "model is %g."%(TRAINING_STEPS,test_acc))

#主程序入口
def main(argv=None):
    #声明处理MNIST数据集的类，这个类在初始化时会自动下载数据
    mnist=input_data.read_data_sets("tmp/data",one_hot=True)
    train(mnist)

#TensorFlow提供的一个主程序入口，tf.app.run()会调用main()函数
if __name__=='__main__':
    tf.app.run()
