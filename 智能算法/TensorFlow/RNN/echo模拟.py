import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epochs=5
total_series_length=50000
truncated_backprop_length=15
state_size=4
num_classes=2
echo_step=3
batch_size=5
num_batches=total_series_length//batch_size//truncated_backprop_length

def generateData():
    #在0和1中选择total_series_length个数
    x=np.array(np.random.choice(2,total_series_length,p=[0.5,0.5]))
    y=np.roll(x,echo_step)#向右循环移位
    y[0:echo_step]=0
    x=x.reshape((batch_size,-1))
    y=y.reshape((batch_size,-1))
    return x,y
def train():
    batchX_placeholder=tf.placeholder(tf.float32,[batch_size,truncated_backprop_length])
    batchY_placeholder=tf.placeholder(tf.int32,[batch_size,truncated_backprop_length])
    init_series=
