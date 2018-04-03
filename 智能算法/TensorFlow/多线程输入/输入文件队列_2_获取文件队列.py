import tensorflow as tf
#通过tf.train.match_filenames_once获取文件列表
filepattern='data.tfrecords*'
files=tf.train.match_filenames_once(pattern=filepattern)
#使用tf.train.string_input_producer创建输入队列
#当一个队列的所有文件都被处理完，会初始化提供的文件列表的文件来重新加入队列
#shuffle参数可以用来随机打乱读文件；num_epochs限制加载初始文件列表的最大轮数
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

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    print(sess.run(files))

    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)

    for i in range(5):
        print(sess.run([features['i'],features['j']]))

    coord.request_stop()
    coord.join()