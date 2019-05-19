import os
import tensorflow as tf
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import conv2d,max_pool2d,fully_connected,avg_pool2d

def random_choose(data,labels,size):
    tot=np.shape(data)[0]
    ids=set([])
    while len(ids)<size:
        i=np.random.randint(0,tot)
        ids.add(i)
    random_data=[]
    random_label=[]
    for i in ids:
        random_data.append(data[i])
        random_label.append(labels[i])
    return random_data,random_label

#配置神经网络参数
BATCH_SIZE=100 #一个训练batch中的训练数据个数。数字越小时，训练过程越接近随机梯度下降；
               #数字越大时，训练越接近梯度下降
LEARNING_RATE_BASE=0.01 #基础的学习率
LEARNING_RATE_DECAY=0.99 #学习率的衰减率
REGULARIZATION_RATE=0.0001 #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS=30000 #训练轮数
MOVING_AVERAGE_DECAY=0.99 #滑动平均衰减率
DISPLAY_SIZE=1000#每1000轮显示一次信息
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
#第二层卷积层的尺寸和深度
CONV3_DEEP=NUM_LABELS
CONV3_SIZE=3
#全连接层的结点个数
FC1_SIZE=512
FC2_SIZE=256
#模型保存路径和文件名
MODEL_SAVE_PATH="model/"
MODEL_NAME="model_of_mnist.ckpt"
#定义网络架构
def inference(input_tensor,train,dropout_rate,regularizer):
    """全连接层使用正则化和dropout_rate"""
    # 第一层：卷积层，28*28*1——>28*28*32，卷积核5*5,0填充，relu
    conv1 = conv2d(input_tensor, CONV1_DEEP, [CONV1_SIZE, CONV1_SIZE], stride=1,
                   padding="SAME",activation_fn=tf.nn.relu,trainable=train)
    #第二层：最大池化层：28*28*32——>14*14*32，过滤器2*2,0填充
    pool1=max_pool2d(conv1,[2,2],stride=2,padding="SAME")
    #第三层：卷积层：14*14*32——>14*14*64,卷积核5*5,0填充，relu
    conv2=conv2d(pool1,CONV2_DEEP,[CONV2_SIZE,CONV2_SIZE],stride=1,padding="SAME",
                 activation_fn=tf.nn.relu,trainable=train)
    #第四层：最大池化层：14*14*64——>7*7*64，过滤器2*2,0填充
    pool2=max_pool2d(conv2,[2,2],stride=2,padding="SAME")
    #第五层：卷积层：7*7*64——>7*7*10，卷积核3*3,0填充，relu
    conv3=conv2d(pool2,CONV3_DEEP,[CONV3_SIZE,CONV3_SIZE],stride=1,padding="SAME",
                 activation_fn=tf.nn.relu,trainable=train)
    #第六层：全局平均池化层：7*7*10——>10，过滤器7*7
    pool3=avg_pool2d(conv3,[7,7],stride=7,padding="SAME")
    return tf.reshape(pool3,[-1,NUM_LABELS])
    #return pool3

def train_eval(mnist):
    #定义输入输出placeholder
    x=tf.placeholder(tf.float32,[BATCH_SIZE,#第一维表示一个batch中样例的个数
                                 IMAGE_SIZE,#第二、三维表示图片尺寸
                                 IMAGE_SIZE,
                                 NUM_CHANNELS],#第四维表示图片的深度，对于RBG图片为5
                     name='x-input')
    y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y =inference(x,True,0.5,regularizer)
    pred=tf.nn.softmax(y)
    global_step = tf.Variable(0, trainable=False)
    #add_global = global_step.assign_add(1)
    # 定义损失函数、学习率、滑动平均操作以及训练过程
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)#采用滑动平均优化算法
    variables_averages_op = variable_averages.apply(tf.trainable_variables())#对所有可训练变量进行滑动平均
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    #cross_entropy=-tf.reduce_sum(y_*tf.log(pred))也适用

    regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = tf.reduce_mean(cross_entropy+regularization_loss) #包含交叉熵损失和正则化损失
    tf.summary.scalar('loss_function', loss)  # 将损失以标量形式显示

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#准确率

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,
                                               mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 在训练神经网络模型时，每过一遍数据需要通过反向传播更新参数，又要更新参数的滑动平均值
    # 为一次完成多次操作，采用tf.control_dependencies和tf.group两种机制
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op=tf.no_op(name='train')
    # 上一句等价于train_op=tf.group(train_step,variables_averages_op)

    # 初始化TensorFlow持久化类
    saver = tf.train.Saver(max_to_keep=5)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        merged_summary_op = tf.summary.merge_all()  # 合并所有summary
        # 创建summary_writer，用于写文件
        summary_writer = tf.summary.FileWriter('log/LeNet5_with_summaries', sess.graph)

        #准备验证数据
        #一般在神经网络训练过程中通过验证数据来大致判断停止的条件和评判训练的结果
        #如果shape中有-1,函数会自动计算维度，但只能有1个-1
        valid_x,valid_y=random_choose(mnist.validation.images,mnist.validation.labels,BATCH_SIZE)
        valid_x=np.reshape(valid_x,(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS))
        validate_feed={x:valid_x,y_:valid_y}
        #准备测试数据
        test_x, test_y = random_choose(mnist.test.images, mnist.test.labels, BATCH_SIZE)
        test_x = np.reshape(test_x, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        test_feed = {x: test_x, y_: test_y}

        for i in range(1, TRAINING_STEPS + 1):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS))

            _,loss_value,step,out,pred_out= sess.run([train_op,loss,global_step,y,pred], feed_dict={x: reshaped_xs, y_: ys})
            #不采用滑动平均模型时，sess.run([train_step, loss, global_step, y, pred], feed_dict={x: reshaped_xs, y_: ys})

            # 生成summary，用于tensorboard
            summary_str = sess.run(merged_summary_op, {x: reshaped_xs, y_: ys})
            summary_writer.add_summary(summary_str, i)  # 将summary(统计信息)写入文件

            # 每1000轮保存一次模型
            if i % DISPLAY_SIZE == 0:
                # 输出当前的训练情况。输出模型在当前batch上的损失函数大小
                #print("out=", out)
                #print("pred_out=", pred_out)
                print("After %d training step(s),loss on training "
                      "batch is %g." % (step, loss_value))

                # 计算模型在验证数据和测试数据上的结果
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                test_acc = sess.run(accuracy, feed_dict=test_feed)
                print("After %d training step(s), validation accuracy using average "
                      "model is %g, test accuracy using average model is %g."
                      % (i, validate_acc, test_acc))
                #保存模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

        # 训练结束后，在测试数据上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average "
                "model is %g." % (TRAINING_STEPS, test_acc))

def evaluate__test(mnist):
    with tf.Graph().as_default() as g:
        #定义输入输出的格式
        x = tf.placeholder(tf.float32, [mnist.test.num_examples,  # 第一维表示一个batch中样例的个数
                                        IMAGE_SIZE,  # 第二、三维表示图片尺寸
                                        IMAGE_SIZE,
                                        NUM_CHANNELS],  # 第四维表示图片的深度，对于RBG图片为5
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
        test_images_reshaped=np.reshape(mnist.test.images,(mnist.test.num_examples,
                                        IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS))
        test_feed={x:test_images_reshaped,y_:mnist.validation.labels}

        #直接通过调用封装好的函数计算前向传播结果
        #测试时不关注正则化损失的值，置为None
        y=inference(x,False,0.5,None)

        #使用前向传播的结果计算正确率
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        #当需要离线预测未知数据的类别时，将该部分改为答案输出

        #通过变量重命名的方式加载模型
        #这样不需要调用求滑动平均值的函数获取平均值，共用一个函数
        variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:
            epoch=26000
            for i in range(5):
                saver.restore(sess,MODEL_SAVE_PATH+MODEL_NAME+"-"+str(epoch))
                global_step = epoch+i*1000
                accuracy_score = sess.run(accuracy, feed_dict=test_feed)
                print("After %s training step(s),test "
                        "accuracy = %.10f" % (global_step, accuracy_score))


def evaluate_validation(mnist):
    with tf.Graph().as_default() as g:
        #定义输入输出的格式
        x = tf.placeholder(tf.float32, [mnist.validation.num_examples,  # 第一维表示一个batch中样例的个数
                                        IMAGE_SIZE,  # 第二、三维表示图片尺寸
                                        IMAGE_SIZE,
                                        NUM_CHANNELS],  # 第四维表示图片的深度，对于RBG图片为5
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
        validation_images_reshaped=np.reshape(mnist.validation.images,(mnist.validation.num_examples,
                                                                       IMAGE_SIZE,
                                                                       IMAGE_SIZE,
                                                                       NUM_CHANNELS))
        validate_feed={x:validation_images_reshaped,y_:mnist.validation.labels}

        #直接通过调用封装好的函数计算前向传播结果
        #测试时不关注正则化损失的值，置为None
        y=inference(x,False,0.5,None)

        #使用前向传播的结果计算正确率
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        #当需要离线预测未知数据的类别时，将该部分改为答案输出

        #通过变量重命名的方式加载模型
        #这样不需要调用求滑动平均值的函数获取平均值，共用一个函数
        variable_averages=tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY
        )
        variables_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:
            epoch=26000
            for i in range(5):
                saver.restore(sess,MODEL_SAVE_PATH+MODEL_NAME+"-"+str(epoch))
                global_step = epoch+i*1000
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print("After %s training step(s),validation "
                        "accuracy = %.10f" % (global_step, accuracy_score))
def main(argv=None):
    mnist=input_data.read_data_sets("tmp/data",one_hot=True)
    train_eval(mnist)
    #evaluate__test(mnist) #GPU不够
    evaluate_validation(mnist)

if __name__=='__main__':
    tf.app.run()