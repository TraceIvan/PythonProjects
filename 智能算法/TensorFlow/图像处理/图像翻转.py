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

    #上下翻转
    flipped_ud=tf.image.flip_up_down(img_data)
    #左右翻转
    flipped_lr=tf.image.flip_left_right(img_data)
    #对角线翻转
    transposed=tf.image.transpose_image(img_data)
    plt.figure(2)
    plt.subplot(1,3,1)
    plt.imshow(flipped_ud.eval())
    plt.subplot(1,3,2)
    plt.imshow(flipped_lr.eval())
    plt.subplot(1,3,3)
    plt.imshow(transposed.eval())
    plt.show()

    #随机图像翻转
    flipped_lr_random=tf.image.random_flip_left_right(img_data)
    flipped_ud_random=tf.image.random_flip_up_down(img_data)
    plt.figure(3)
    plt.subplot(1,2,1)
    plt.imshow(flipped_ud_random.eval())
    plt.subplot(1,2,2)
    plt.imshow(flipped_lr_random.eval())
    plt.show()
