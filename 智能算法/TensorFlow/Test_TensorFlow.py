import tensorflow as tf
w1=tf.Variable(tf.random_normal([2,3],stddev=1),name="w1")
w2=tf.Variable(tf.random_normal([2,2],stddev=1),name="w2")


sess=tf.Session()
sess.run(w1.initializer)
sess.run(w2.initializer)
print(sess.run(w1))
tf.assign(w2,w1,validate_shape=False)
print(sess.run(w1))
print(sess.run(w2))
sess.close()