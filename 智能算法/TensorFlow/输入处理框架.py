import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#创建文件列表，并创建输入文件队列
files=tf.train.match_filenames_once('*.tfrecords')
filename_queue=tf.train.string_input_producer(files,shuffle=False)

#解析TFRecord文件数据，假设image为图像原始数据，label为对应标签，height、width和channels给出图片的维度
reader=tf.TFRecordReader()
_,serialized_example=reader.read(filename_queue)
features=tf.parse_single_example(
    serialized_example,
    features={
        'image':tf.FixedLenFeature([],tf.string),
        'label':tf.FixedLenFeature([],tf.int64),
        'height':tf.FixedLenFeature([],tf.int64),
        'width':tf.FixedLenFeature([],tf.int64),
        'channels':tf.FixedLenFeature([],tf.int64)
    })
image,label=features['image'],features['label']
height,width=features['height'],features['width']
channels=features['channels']

#从原始图像数据解析出像素矩阵，并根据图像尺寸还原图像
decoded_image=tf.decode_raw(image,tf.uint8)
decoded_image.set_shape([height,width,channels])

#定义神经网络输入层图片大小
image_size=300

#图像预处理

#给定一张图片，随机调整图像的色彩
#调整亮度、对比度、饱和度和色相的顺序影响最后的结果，可定义多种不同的顺序
def distort_color(image,color_ordering=0):
    if color_ordering==0:
        image=tf.image.random_brightness(image,max_delta=32./255.)
        image=tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image=tf.image.random_hue(image,max_delta=0.2)
        image=tf.image.random_contrast(image,lower=0.5,upper=1.5)
    elif color_ordering==1:
        image=tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering==2:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    return tf.clip_by_value(image,0.0,1.0)
#给定一张解码后的图像、目标图像的尺寸以及图像上的标注框
def preprocess_for_train(image,height,width,bbox):
    if bbox is None:
        bbox=tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4])
    if image.dtype != tf.float32:
        image=tf.image.convert_image_dtype(image,dtype=tf.float32)
    bbox_begin,bbox_size,_=tf.image.sample_distorted_bounding_box(tf.shape(image),bounding_boxes=bbox)
    distorted_image=tf.slice(image,bbox_begin,bbox_size)
    distorted_image=tf.image.resize_images(
        distorted_image,[height,width],method=np.random.randint(4)
    )
    distorted_image=tf.image.random_flip_left_right(distorted_image)
    distorted_image=tf.image.random_flip_up_down(distorted_image)
    distorted_image=distort_color(distorted_image,np.random.randint(3))
    return distorted_image

#预处理结果
distort_image=preprocess_for_train(decoded_image,image_size,image_size,None)

#将处理后的图像和标签通过tf.train.shuffle_batch整理成需要的batch
min_after_dequeue=10000
batch_size=100
capacity=min_after_dequeue+3*batch_size
image_batch,label_batch=tf.train.shuffle_batch([distort_image,label],batch_size=batch_size,
                                               capacity=capacity,min_after_dequeue=min_after_dequeue)

#定义神经网络的结构和优化过程(简化)
#image_batch提供给输入层，label_batch提供输入层的样例的正确答案
logit=inference(image_batch)
loss=calc_loss(logit,label_batch)
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#声明会话并运行神经网络的优化过程
with tf.Session() as sess:
    #训练准备阶段，包括变量初始化，线程启动
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)

    #训练过程
    for i in range(TRAINING_ROUNDS):
        sess.run(train_step)

    #停止所有线程
    coord.request_stop()
    coord.join(threads)