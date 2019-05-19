import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
a = tf.test.is_built_with_cuda()  # 判断CUDA是否可以用
b = tf.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None) # 判断GPU是否可以用
print(a)
print(b)

plotdata={"batchsize":[],"loss":[]}#存放批次和对应损失值
def moving_average(a,w=10):
    if len(a)<w:
        return a[:]
    return [val if idx <w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]

#1、准备数据
train_X=np.linspace(-1,1,100)
train_Y=2*train_X+np.random.randn(*train_X.shape)*0.3 #y=2x，并加入噪声
#显示模拟数据
plt.plot(train_X,train_Y,'ro',label='Original data')
plt.legend()
plt.show()

#2、创建模型
tf.reset_default_graph()
#占位符
X=tf.placeholder("float")
Y=tf.placeholder("float")
#模型参数
W=tf.Variable(tf.random_normal([1]),name="weight")
b=tf.Variable(tf.zeros([1]),name="bias")
#前向结构
z=tf.multiply(X,W)+b#点乘
tf.summary.histogram('z',z) #将预测值以直方图形式显示
#反向优化
cost=tf.reduce_mean(tf.square(Y-z))
tf.summary.scalar('loss_function',cost)#将损失以标量形式显示
learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)#梯度下降

#3、训练模型
#初始化所有变量
init=tf.global_variables_initializer()
#定义参数
training_epochs=20 #迭代次数
display_step=2
saver=tf.train.Saver()
saver2=tf.train.Saver(max_to_keep=1)#生成saver
savedir="log/"#生成模型的路径

#启动session
with tf.Session() as sess:
    sess.run(init)
    merged_summary_op=tf.summary.merge_all()#合并所有summary
    #创建summary_writer，用于写文件
    summary_writer=tf.summary.FileWriter('log/linermodel_with_summaries',sess.graph)
    #向模型输入数据
    for epoch in range(training_epochs):
        for(x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
            # 生成summary
            summary_str = sess.run(merged_summary_op, feed_dict={X: x, Y: y})
            summary_writer.add_summary(summary_str,epoch)#将summary写入文件

        #显示训练中的信息
        if epoch % display_step ==0:
            loss=sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            print("Epoch:",epoch+1,"cost=",loss,"W=",sess.run(W),"b=",sess.run(b))
            if not (loss=="NA"):
                plotdata["batchsize"].append(epoch)
                plotdata['loss'].append(loss)
            saver2.save(sess,savedir+'linermodel.cpkt',global_step=epoch)



    print(" Finished!")
    saver.save(sess,savedir+'linermodel.cpkt')#保存模型
    from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    print_tensors_in_checkpoint_file(savedir+"linermodel.cpkt",None,True)#查看模型内容
    print("cost=",sess.run(cost,feed_dict={X:train_X,Y:train_Y}),"W=",sess.run(W),"b=",sess.run(b))



    #图形显示
    plt.plot(train_X,train_Y,'ro',label='Original data')
    plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label='Fittedline')
    plt.legend()
    plt.show()

    plotdata["avgloss"]=moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"],plotdata["avgloss"],'b--')
    plt.xlabel("Minibatch number")
    plt.ylabel("Loss")
    plt.title("Minibatch run vs. Training Avg Loss")
    plt.show()
#载入线性模型
with tf.Session() as sess2:
    sess2.run(init)
    saver.restore(sess2,savedir+'linermodel.cpkt')
    print("x=0.2,z=",sess2.run(z,feed_dict={X:0.2}))

#载入指定检查点的模型
load_epoch=18
with tf.Session() as sess3:
    sess3.run(init)
    saver2.restore(sess3,savedir+'linermodel.cpkt-'+str(load_epoch))
    print("x=0.2,z=",sess3.run(z,feed_dict={X:0.2}))

#载入所有检查点文件
with tf.Session() as sess4:
    sess4.run(init)
    ckpt=tf.train.get_checkpoint_state(savedir)
    if ckpt and ckpt.model_checkpoint_path:
        saver2.restore(sess4,ckpt.model_checkpoint_path)
        print("x=0.2,z=",sess4.run(z,feed_dict={X:0.2}))

