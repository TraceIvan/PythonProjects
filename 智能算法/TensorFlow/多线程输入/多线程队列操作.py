import tensorflow as tf

#声明一个先进先出的队列，队列中最多有100个元素，类型为实数
queue=tf.FIFOQueue(100,"float")
#定义队列的入队操作
enqueue_op=queue.enqueue([tf.random_normal([1])])

#使用tf.train.QueueRunner来创建多个线程运行队列的入队操作
#参数：被操作队列；[enqueue_op]*5表示需要启动5个线程，都运行enqueue_op操作
qr=tf.train.QueueRunner(queue,[enqueue_op]*5)

#将定义过的QueueRunner加入tf计算图上指定的集合
#tf.train.add_queue_runner()函数没有指定集合时则加入默认的tf.GraphKeys.QUEUE_RUNNERS
tf.train.add_queue_runner(qr)
#定义出队操作
out_tensor=queue.dequeue()

with tf.Session() as sess:
    coord=tf.train.Coordinator()#协同启动的线程
    #使用tf.train.add_queue_runner()需要明确调用tf.train.start_queue_runners来启动所有线程
    #否则因为没有线程运行入队操作，当调用出队操作时，程序会一直等待入队操作被执行
    #tf.train.start_queue_runners会默认启动tf.GraphKeys.QUEUE_RUNNERS集合中所有的QueueRunner
    #一般而言tf.train.add_queue_runner和tf.train.start_queue_runners会指定同一个集合
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    #获取队列中的值
    for _ in range(8):
        print(sess.run(out_tensor))
    #使用tf.train.Coordinator停止所有进程
    coord.request_stop()
    coord.join(threads)

