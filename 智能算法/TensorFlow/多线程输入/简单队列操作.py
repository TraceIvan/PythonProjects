import tensorflow as tf

#创建一个先进先出的队列，指定队列中最多保存两个元素，指定类型为整数
q=tf.FIFOQueue(2,"int32")
#使用enqueue_many函数初始化队列元素
init=q.enqueue_many(([0,10],))
#使用Dequeue函数将队列的第一个元素出队列，值存在x变量中
x=q.dequeue()
#将得到的值+1
y=x+1
#将加1后的值重新加入队列
q_inc=q.enqueue([y])

with tf.Session() as sess:
    #运行初始化队列操作
    init.run()
    for _ in range(5):
        #运行q_inc执行数据出队列、出队元素+1、重新加入队列的整个过程
        v,_=sess.run([x,q_inc])
        #打印出队元素的取值
        print(v)