import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

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
#输入为原始的训练图像，输出为神经网络的输入层
#只处理模型的训练数据，预测数据一般不需要随机变换的步骤
def preprocess_for_train(image,height,width,bbox):
    #如果没有提供提供标注框，则认为需要关注的部分为整个图像
    if bbox is None:
        bbox=tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4])
    #转换图像张亮的类型
    if image.dtype != tf.float32:
        image=tf.image.convert_image_dtype(image,dtype=tf.float32)

    #随机截取图像，减小需要关注的物体大小对图像识别算法的影响
    bbox_begin,bbox_size,_=tf.image.sample_distorted_bounding_box(tf.shape(image),bounding_boxes=bbox)
    distorted_image=tf.slice(image,bbox_begin,bbox_size)
    #将随机截取的图片调整为输入层的大小
    distorted_image=tf.image.resize_images(
        distorted_image,[height,width],method=np.random.randint(4)
    )
    #随机左右翻转图像
    distorted_image=tf.image.random_flip_left_right(distorted_image)
    #随机上下翻转图像
    distorted_image=tf.image.random_flip_up_down(distorted_image)
    #使用一种随机的顺序调整图像色彩
    distorted_image=distort_color(distorted_image,np.random.randint(3))

    return distorted_image

image_raw_data=tf.gfile.FastGFile("picture1.jpg","rb").read()
with tf.Session() as sess:
    img_data=tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.1, 0.15, 0.7, 0.5], [0.39, 0.28, 0.51, 0.36]]])

    #运行6次获得6种不同的图像
    plt.figure(1)
    for i in range(6):
        #将图像的尺寸调整为200*350
        result=preprocess_for_train(img_data,200,400,boxes)
        plt.subplot(3,2,i+1)
        plt.imshow(result.eval())
    plt.show()