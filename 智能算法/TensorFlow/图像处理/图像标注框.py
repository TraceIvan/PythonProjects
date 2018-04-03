import matplotlib.pyplot as plt
import tensorflow as tf

#读取图像原始数据
image_raw_data=tf.gfile.FastGFile("picture1.jpg",'rb').read()

with tf.Session() as sess:
    #将图片采用JPEG的格式解码，得到对应三维矩阵
    #tf.image.decode_png用来对png格式的图像解码
    #解码结果为张量
    img_data=tf.image.decode_jpeg(image_raw_data)
    print(img_data.eval())
    #使用pyplot工具可视化
    plt.figure(1)
    plt.imshow(img_data.eval())
    plt.show()
    #将数据类型转换为实数方便图像处理
    img_data=tf.image.convert_image_dtype(img_data,dtype=tf.float32)

    #先缩小图片，可视化时标注框更加清楚
    img_data=tf.image.resize_images(img_data,[200,350],method=3)
    #tf.image.draw_bounding_boxes要求图像矩阵数字为实数
    #输入是一个batch，加一维
    batched=tf.expand_dims(tf.image.convert_image_dtype(img_data,dtype=tf.float32),0)
    #给出一个batch所有图像的标注框（这里只有一张）
    #【Ymin,Xmin,Ymax,Xmax】为0~1之间的小数，为相对位置
    boxes=tf.constant([[[0.1,0.15,0.7,0.5],[0.39,0.28,0.51,0.36]]])
    result=tf.image.draw_bounding_boxes(batched,boxes)
    plt.figure(2)
    plt.imshow(result.eval()[0,:,:,:])
    plt.show()

    #随机截取图像上有信息含量的部分（使模型不受识别物体大小影响）
    begin,size,bbox_for_draw=tf.image.sample_distorted_bounding_box(tf.shape(img_data),bounding_boxes=boxes)
    #通过标注框可视化随机截取的图像
    batched_random=tf.expand_dims(img_data,0)
    image_with_box=tf.image.draw_bounding_boxes(batched_random,bbox_for_draw)
    plt.figure(3)
    plt.imshow(image_with_box.eval()[0,:,:,:])
    plt.show()
    #截取随机出来的图像
    distorted_image=tf.slice(img_data,begin=begin,size=size)
    plt.figure(4)
    plt.imshow(distorted_image.eval())
    plt.show()