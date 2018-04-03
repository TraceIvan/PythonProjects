# -*- coding: utf-8 -*-
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#加载mnist_inference.py中定义的常量和前向传播函数
import mnist_inference

#配置神经网络参数
BATCH_SIZE=100 #一个训练batch中的训练数据个数。数字越小时，训练过程越接近随机梯度下降；
               #数字越大时，训练越接近梯度下降
LEARNING_RATE_BASE=0.8 #基础的学习率
LEARNING_RATE_DECAY=0.99 #学习率的衰减率
REGULARIZATION_RATE=0.0001 #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS=30000 #训练轮数
MOVING_AVERAGE_DECAY=0.99 #滑动平均衰减率

#模型保存路径和文件名
MODEL_SAVE_PATH="model/"
MODEL_NAME="model_of_mnist.ckpt"

def train(mnist):
    #定义输入输出placeholder
    x=tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name='x-input')
    y_=tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name='y-input')

    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #直接使用mnist_inference.py定义的前向传播过程
    y=mnist_inference.inference(x,regularizer)
    global_step=tf.Variable(0,trainable=False)

    #定义损失函数、学习率、滑动平均操作以及训练过程
    variable_averages=tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY,global_step
    )
    variables_averages_op=variable_averages.apply(
        tf.trainable_variables()
    )
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    learning_rate=tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op=tf.no_op(name='train')

    #初始化TensorFlow持久化类
    saver=tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        #在训练过程中不再测试模型在验证数据上的表现
        for i in range(1,TRAINING_STEPS+1):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            # 每1000轮保存一次模型
            if i % 1000 == 0:
                # 输出当前的训练情况。此处只输出模型在当前batch上的损失函数大小
                print("After %d training step(s),loss on training "
                      "batch is %g." % (step, loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main(argv=None):
    mnist=input_data.read_data_sets("../tmp/data",one_hot=True)
    train(mnist)

if __name__=='__main__':
    tf.app.run()