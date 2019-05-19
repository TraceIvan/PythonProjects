import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from utils import *
class batch_norm(object):
    '''批标准化'''
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.name = name

    def __call__(self, x, train):#转成可调用对象
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon,
                                            center=True, scale=True, is_training=train, scope=self.name)

def binary_cross_entropy(preds, targets, name=None):
    '''计算交叉熵'''
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +(1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    '''连接两个矩阵'''
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])],3)

def conv2d(input_, output_dim,k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,name="conv2d"):
    '''定义卷积层'''
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME') #默认大小缩小一倍
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        return conv

def conv2d_transpose(input_, output_shape,k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,name="conv2d_transpose", with_w=False):
    '''定义小步长卷积（又称上卷积、转置卷积）'''
    with tf.variable_scope(name):
        # 卷积核: [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        # tf 0.7版本之前使用：tf.nn.deconv2d
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))

        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:#返回权值和偏置
            return deconv, w, biases
        else:
            return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    '''定义激活函数LReLU'''
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    '''定义线性函数，input_为2维，[batch_size,size]'''
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

if __name__=='__main__':
    x=tf.constant([[[[1.0,2],[3,4]],[[5,6],[7,8]]],[[[9,10],[11,12]],[[13,14],[15,16]]]])
    y=tf.constant([[[[17.0,18],[19,20]],[[21,22],[23,24]]],[[[25,26],[27,28]],[[29,30],[31,32]]]])
    sess=tf.Session()
    print(sess.run(conv_cond_concat(x,y)))