import tensorflow as tf
#通过tf.train.match_filenames_once获取文件列表
filepattern='data.tfrecords*'
files=tf.train.match_filenames_once(pattern=filepattern)
#使用tf.train.string_input_producer创建输入队列
#shuffle参数可以用来随机打乱读文件
filename_queue=tf.train.string_input_producer(files,shuffle=False)

#读取并解析一个样本
reader=tf.TFRecordReader()
_,serialized_example=reader.read(filename_queue)
features=tf.parse_single_example(
    serialized_example,
    features={
    'i':tf.FixedLenFeature([],tf.int64),
    'j':tf.FixedLenFeature([],tf.int64)
    })
#假设example结构中i表示一个样例的特征向量，比如一张图片的像素矩阵，j对应标签
example,label=features['i'],features['j']

#一个batch_size中的样例的个数
batch_size=3
#组合样例的队列中最多可以存储的样例个数
#太大会占用很多内存资源；太小会使出队操作因没有数据而被阻塞
capacity=1000+3*batch_size
#当队列长度等于容量时，tf将暂停入队操作，只是等待元素出队
#当元素个数小于容量时，tf将自动重启入队操作
#example_batch,label_batch=tf.train.batch([example,label],batch_size=batch_size,capacity=capacity)
#tf.train.shuffle_batch通过min_after_dequeue限制出队时队列中元素的最小个数以满足随机打乱顺序的作用
example_batch,label_batch=tf.train.shuffle_batch([example,label],batch_size=batch_size,capacity=capacity,min_after_dequeue=30)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    sess.run(files)
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    #获取并打印组合之后的样例。在真实问题中，这个输出一般会作为神经网络的输入
    for i in range(2):
        sess.run(features)
        cur_example_batch,cur_label_batch=sess.run([example_batch,label_batch])
        print(cur_example_batch,cur_label_batch)
    coord.request_stop()
    coord.join(threads)