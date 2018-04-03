import  tensorflow as tf
filename="tmp/output.tfrecords"
#创建一个reader读取TFRecord文件中的样例
reader=tf.TFRecordReader()
#创建一个队列维护输入文件列表
filename_queue=tf.train.string_input_producer([filename])

#从文件中读取一个样例
#read_up_to可以读取多个样例
_,serialized_example=reader.read(filename_queue)

#解析读入的一个样例
#解析多个样例：parse_example
features=tf.parse_single_example(
    serialized_example,
    features={
        #两种属性解析方法
        #tf.FixedLenFeature:结果为Tensor
        #tf.VarLenFeature:结果为SparseTensor,用于处理稀疏数据
        'image_raw':tf.FixedLenFeature([],tf.string),
        'pixels':tf.FixedLenFeature([],tf.int64),
        'label':tf.FixedLenFeature([],tf.int64),
    }
)
#tf.decode_raw可以将字符串解析成图像对应的像素数组
images=tf.decode_raw(features['image_raw'],tf.uint8)
labels=tf.cast(features['label'],tf.int32)
pixels=tf.cast(features['pixels'],tf.int32)

sess=tf.Session()
#启动多线程处理输入数据
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=sess,coord=coord)

#每次运行可以读取一个样例
#所有样例读完后，在此程序中会重头读取
for i in range(10):
    image,label,pixel=sess.run([images,labels,pixels])
    print(image)
    print([label,pixel])
