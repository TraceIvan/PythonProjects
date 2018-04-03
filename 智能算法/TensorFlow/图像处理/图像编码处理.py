#matplotlib.pyplot是一个pyhton的画图工具
#此处用来可视化经过TensorFlow处理的图像
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
    plt.imshow(img_data.eval())
    plt.show()
    #将数据类型转换为实数方便图像处理
    #img_data=tf.image.convert_image_dtype(img_data,dtype=tf.float32)
    #print(img_data.eval())

    #将表示一张图像的三维矩阵重新按照jpeg格式编码存入文件
    encode_image=tf.image.encode_jpeg(img_data)#转换成实数后不能再编码
    with tf.gfile.GFile("output.jpeg","wb") as f:
        f.write(encode_image.eval())
