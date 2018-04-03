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

    #通过tf.image.resize_images函数调整图像大小
    #第一个参数为原始图像，第二个参数、第三个参数为调整后的图像大小，method为图像调整算法
    resized_0=tf.image.resize_images(img_data,[300,300],method=0)#双线性插值法
    resized_1 = tf.image.resize_images(img_data, [300, 300], method=1)#最近邻居法
    resized_2 = tf.image.resize_images(img_data, [300, 300], method=2)#双三次插值法
    resized_3 = tf.image.resize_images(img_data, [300, 300], method=3)#面积插值法

    #输出调整后的图像大小
    #图像深度在没有明确设置之前是问号
    print(resized_0.get_shape())
    #plt.imshow(resized_0.eval())
    plt.figure(2)
    plt.subplot(2,2,1)
    plt.imshow(resized_0.eval())
    plt.subplot(2, 2, 2)
    plt.imshow(resized_1.eval())
    plt.subplot(2, 2, 3)
    plt.imshow(resized_2.eval())
    plt.subplot(2, 2, 4)
    plt.imshow(resized_3.eval())
    plt.show()

    #tf.image.resize_image_with_crop_or_pad调整图像大小
    #参数：原始图像；目标图像长、宽
    #如果目标图像大于原始图像，自动填充全0背景；如果小于，自动截取居中部分
    croped=tf.image.resize_image_with_crop_or_pad(img_data,1000,1000)
    padded=tf.image.resize_image_with_crop_or_pad(img_data,3000,3000)
    plt.figure(3)
    plt.subplot(1,2,1)
    plt.imshow(croped.eval())
    plt.subplot(1,2,2)
    plt.imshow(padded.eval())
    plt.show()

    #tf.image.central_crop按比例中央裁剪图像
    central_cropped=tf.image.central_crop(img_data,0.5)
    plt.figure(4)
    plt.imshow(central_cropped.eval())
    plt.show()

    #tf.image.crop_to_bounding_box画框裁剪
    #{起点高度，起点宽度，框高，框宽}
    croped_box=tf.image.crop_to_bounding_box(img_data,100,100,500,500)
    plt.figure(5)
    plt.imshow(croped_box.eval())
    plt.show()

    #tf.image.pad_to_bounding_box() 用0填充边界，使输入图像符合期望尺寸。尺寸过大过小图像，边界填充灰度值0像素。
    padded_box=tf.image.pad_to_bounding_box(img_data,100,100,1500,2500)
    plt.figure(6)
    plt.imshow(padded_box.eval())
    plt.show()
