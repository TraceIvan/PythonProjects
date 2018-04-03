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

    #亮度-0.5
    adjusted_1=tf.image.adjust_brightness(img_data,-0.5)
    #亮度+0.5
    adjusted_2=tf.image.adjust_brightness(img_data,0.5)
    #在[-max_delta,max_delta)随机调整亮度
    max_delta=0.8
    adjusted_3=tf.image.random_brightness(img_data,max_delta)
    plt.figure(2)
    plt.subplot(1,3,1)
    plt.imshow(adjusted_1.eval())
    plt.subplot(1,3,2)
    plt.imshow(adjusted_2.eval())
    plt.subplot(1,3,3)
    plt.imshow(adjusted_3.eval())
    plt.show()

    #对比度-5
    adjusted_contrast_1=tf.image.adjust_contrast(img_data,-5)
    #对比度+5
    adjusted_contrast_2=tf.image.adjust_contrast(img_data,5)
    #[lower,upper]随机调整对比度
    lower=2
    upper=9
    adjusted_contrast_3=tf.image.random_contrast(img_data,lower,upper)
    plt.figure(3)
    plt.subplot(1,3,1)
    plt.imshow(adjusted_contrast_1.eval())
    plt.subplot(1,3,2)
    plt.imshow(adjusted_contrast_2.eval())
    plt.subplot(1,3,3)
    plt.imshow(adjusted_contrast_3.eval())
    plt.show()

    #色相加0.1,0.3,0.6,0.9
    adjusted_hue_1=tf.image.adjust_hue(img_data,0.1)
    adjusted_hue_2=tf.image.adjust_hue(img_data,0.3)
    adjusted_hue_3=tf.image.adjust_hue(img_data,0.6)
    adjusted_hue_4=tf.image.adjust_hue(img_data,0.9)
    #在[-max_delta,max_delta]随机调整
    max_delta=0.3
    adjusted_hue_5=tf.image.random_hue(img_data,max_delta)
    plt.figure(4)
    plt.subplot(3,2,1)
    plt.imshow(adjusted_hue_1.eval())
    plt.subplot(3,2,2)
    plt.imshow(adjusted_hue_2.eval())
    plt.subplot(3,2,3)
    plt.imshow(adjusted_hue_3.eval())
    plt.subplot(3,2,4)
    plt.imshow(adjusted_hue_4.eval())
    plt.subplot(3,1,3)
    plt.imshow(adjusted_hue_5.eval())
    plt.show()

    #饱和度+5
    adjusted_saturation_1=tf.image.adjust_saturation(img_data,5)
    #饱和度-5
    adjusted_saturation_2=tf.image.adjust_saturation(img_data,-5)
    #[lower,upper]随机调整
    lower=1.2
    upper=6.8
    adjusted_saturation_3=tf.image.random_saturation(img_data,lower,upper)
    plt.figure(5)
    plt.subplot(3,1,1)
    plt.imshow(adjusted_saturation_1.eval())
    plt.subplot(3,1,2)
    plt.imshow(adjusted_saturation_2.eval())
    plt.subplot(3,1,3)
    plt.imshow(adjusted_saturation_3.eval())
    plt.show()

    #图像标准化
    #数字均值变为0，方差变为1
    adjusted_whiten=tf.image.per_image_standardization(img_data)
    plt.figure(6)
    plt.imshow(adjusted_whiten.eval())
    plt.show()