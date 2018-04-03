import tensorflow as tf
a=tf.constant([1.0,2.0],name="a")#定义常量
b=tf.constant([2.0,3.0],name="b")
result=a+b
sess=tf.Session()#生成会话
print(sess.run(result))