import tensorflow as tf
if __name__=='__main__':
    x=tf.constant([[1.1,2.2,3.3,4.4],[5.5,6.6,7.7,8.8]])#[2,4]
    print(x.shape)
    m=tf.constant([3])
    k=tf.constant([4])
    d = x.get_shape().as_list()[-1]
    print(d)
    W = tf.Variable(tf.random_normal(shape=[d, 3, 4], stddev=0.1))#[4,3,4]
    b = tf.Variable(tf.random_normal(shape=[3, 4], stddev=0.1))
    z1 = tf.tensordot(x, W, axes=1) + b#[2,3,4]
    z2 = tf.reduce_max(z1, axis=2)#[2,3]
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        print(sess.run(W))
        print(sess.run(b))
        print(sess.run(z1))
        print(sess.run(z2))