# -*- coding: utf-8 -*-
import tensorflow as tf

#配置神经网络参数
INPUT_NODE=784
OUTPUT_NODE=10

IMAGE_SIZE=28
NUM_CHANNELS=1
NUM_LABELS=10

#第一层卷积层的尺寸和深度
CONV1_DEEP=32
CONV1_SIZE=5
#第二层卷积层的尺寸和深度
CONV2_DEEP=64
CONV2_SIZE=5
#全连接层的结点个数
FC_SIZE=512

#定义卷积神经网络的前向传播过程
#train参数区分训练过程和测试过程
def inference(input_tensor,train,regularizer):
    #声明第一层卷积层的变量并实现前向传播过程
    #输入为28*28*1的原始MNIST图片像素，使用全0填充，输出为28*28*32矩阵
    with tf.variable_scope('layer1-conv1',reuse=tf.AUTO_REUSE):
        conv1_weights=tf.get_variable(
            "weights",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )#前两个维度表示过滤器尺寸，第三个为当前层的深度，第四个为过滤器的深度
        conv1_biases=tf.get_variable(
            "biases",[CONV1_DEEP],initializer=tf.constant_initializer(0.1)
        )
        #使用边长为5，深度为32的过滤器，移动步长为1，使用全0填充
        #参数：当前层的节点矩阵；卷积层的权重；不同维度上的步长；填充方法
        conv1=tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        #给每一个结点加上同样的偏置项
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    #第二层迟化池前向传播过程
    #选用：最大迟化池；过滤器边长为2，使用全0填充，移动步长为2
    #输入为28*28*32矩阵，输出为14*14*32矩阵
    with tf.name_scope('layer2-pool1'):
        pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        #参数：当前层的节点矩阵；过滤器的尺寸[一个输入batch(样例);长;宽;深度](
        # [1,2,2,1]、[1,3,3,1]);步长[1,X,X,1]；填充方法

    #声明第三层卷积层变量并前向传播
    #输入为14*14*32，输出为14*14*64
    with tf.variable_scope('layer3-conv2',reuse=tf.AUTO_REUSE):
        conv2_weights=tf.get_variable(
            "weights",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv2_biases=tf.get_variable(
            "biases",[CONV2_DEEP],initializer=tf.constant_initializer(0.1)
        )
        #使用边长为5，深度为64的过滤器，移动步长为1，使用全0填充
        conv2=tf.nn.conv2d(
            pool1,conv2_weights,strides=[1,1,1,1],padding='SAME'
        )
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    #第四层池化层前向传播
    #输入为14*14*64矩阵，输出为7*7*64矩阵
    with tf.name_scope('layer4-pool2'):
        pool2=tf.nn.max_pool(
            relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'
        )

    #将第四层池化层输出转化为第五层全连接层的输入格式
    #将7*7*64矩阵拉直成一个向量
    #pool2.get_shape可以得到维度
    #因为每一层输入输出为一个batch的矩阵，这里的维度也包含一个batch中数据的个数
    pool_shape=pool2.get_shape().as_list()
    #计算拉直后向量的长度。pool_shape[0]为一个batch中数据的个数
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
    #通过tf.reshape将第四层的输出变为一个batch的向量
    reshaped=tf.reshape(pool2,[pool_shape[0],nodes])

    #声明第五层全连接层的变量并实现前向传播过程,输出512长度的向量
    #dropout在训练时会随机将部分节点输出改为0，避免过度拟合问题
    #dropout一般只在全连接层而不是卷积层或池化层使用
    with tf.variable_scope('layer5-fc1',reuse=tf.AUTO_REUSE):
        fc1_weights=tf.get_variable(
            "weights",[nodes,FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        #只有全连接层的权重需要加入正则化
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases=tf.get_variable(
            "biases",[FC_SIZE],initializer=tf.constant_initializer(0.1)
        )
        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        if train:
            fc1=tf.nn.dropout(fc1,0.5)

    #声明第六层全连接层的变量并实现前向传播
    #输入为512长度向量，输出为长度为10的向量
    #这一层的输出通过softmax得到最后分类结果
    with tf.variable_scope('layer6-fc2',reuse=tf.AUTO_REUSE):
        fc2_weights=tf.get_variable(
            "weights",[FC_SIZE,NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases=tf.get_variable(
            "biases",[NUM_LABELS],
            initializer=tf.constant_initializer(0.1)
        )
        logit=tf.matmul(fc1,fc2_weights)+fc2_biases

    #返回第六层输出
    return logit
