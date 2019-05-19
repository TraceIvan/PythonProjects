import base
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import ctc_ops
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import keras
from collections import Counter


wav_path="E:/迅雷下载/CSLT public data/thchs30-standalone/wav/train"
label_file="E:/迅雷下载/CSLT public data/thchs30-standalone/doc/trans/train.word.txt"
wav_files,labels=base.get_wavs_labels(wav_path,label_file)
ALL_WORDS=[]#字表
for label in labels:
    ALL_WORDS+=[word for word in label]
counter=Counter(ALL_WORDS)
words=sorted(counter)
words_size=len(words)
word_num_map=dict(zip(words,range(words_size)))
print('字表大小：',words_size)

#参数设置
n_input=26 #计算MFCC的个数
n_context=9 #对于每个时间点，包含上下文样本(时序)的个数
batch_size=8

#取下一批次数据
def next_batch(labels,start_idx=0,batch_size=1,wav_files=wav_files):
    filesize=len(labels)
    end_idx=min(filesize,start_idx+batch_size)
    idx_list=range(start_idx,end_idx)
    txt_labels=[labels[i] for i in idx_list]
    wav_files=[wav_files[i] for i in idx_list]
    source,audio_len,target,transcript_len=base.get_audio_and_transcriptch(None,wav_files,n_input,n_context,
                                                                           word_num_map,txt_labels)
    start_idx+=batch_size
    if start_idx>=filesize:
        start_idx=-1
    source,source_lengths=base.pad_sequences(source)#时序长度统一
    sparse_labels=base.sparse_tuple_from(target)#文本转为稀疏矩阵
    return start_idx,source,source_lengths,sparse_labels

def test1():
    next_idx,source,source_len,sparse_lab=next_batch(labels,0,batch_size)
    print(len(sparse_lab))
    print(np.shape(source))
    t=base.sparse_tuple_to_texts_ch(sparse_lab,words)
    print(t[0])


tf.reset_default_graph()
#BiRNN参数
b_stddev=0.046875
w_stddev=0.046875
n_hidden=512
n_hidden_1=512#第一层，全连接层
n_hidden_2=512#第二层，全连接层
n_hidden_3=2*512#第三层，全连接层
n_hidden_5=512
n_cell_dim=512
keep_dropout_rate=0.95
REGULARIZATION_RATE=0.0001 #描述模型复杂度的正则化项在损失函数中的系数
#relu_clip=20
savedir = "log/"

#定义网络架构
def inference(input_tensor,n_character,train,dropout_rate,regularizer,seq_length):
    batch_x_shape=tf.shape(input_tensor)
    #将输入转成时间序列优先
    batch_x=tf.transpose(input_tensor,[1,0,2])
    #转成2维,(n_steps*batch_size,n_input+2*n_input*n_context)
    batch_x=tf.reshape(batch_x,[-1,n_input+2*n_input*n_context])

    #第一层，全连接层，输入n_input+2*n_input*n_context，输出n_hidden_1
    with tf.variable_scope('layer1-fc1',reuse=tf.AUTO_REUSE):
        fc1_weights=tf.get_variable(
            "weights",[n_input+2*n_input*n_context,n_hidden_1],
            initializer=tf.random_normal_initializer(stddev=0.046875)
        )
        #只有全连接层的权重需要加入正则化
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases=tf.get_variable(
            "biases",[n_hidden_1],initializer=tf.random_normal_initializer(stddev=0.046875)
        )
        fc1=tf.nn.relu(tf.matmul(batch_x,fc1_weights)+fc1_biases)
        if train:
            fc1=tf.nn.dropout(fc1,dropout_rate)

    #第二层，全连接层，输入n_hidden_1，输出n_hidden_2
    with tf.variable_scope('layer2-fc2',reuse=tf.AUTO_REUSE):
        fc2_weights=tf.get_variable(
            "weights",[n_hidden_1,n_hidden_2],
            initializer=tf.random_normal_initializer(stddev=0.046875)
        )
        #只有全连接层的权重需要加入正则化
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases=tf.get_variable(
            "biases",[n_hidden_2],initializer=tf.random_normal_initializer(stddev=0.046875)
        )
        fc2=tf.nn.relu(tf.matmul(fc1,fc2_weights)+fc2_biases)
        if train:
            fc2=tf.nn.dropout(fc2,dropout_rate)

    #第三层，全连接层，输入n_hidden_2，输出n_hidden_3
    with tf.variable_scope('layer3-fc3',reuse=tf.AUTO_REUSE):
        fc3_weights=tf.get_variable(
            "weights",[n_hidden_2,n_hidden_3],
            initializer=tf.random_normal_initializer(stddev=0.046875)
        )
        #只有全连接层的权重需要加入正则化
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(fc3_weights))
        fc3_biases=tf.get_variable(
            "biases",[n_hidden_3],initializer=tf.random_normal_initializer(stddev=b_stddev)
        )
        fc3=tf.nn.relu(tf.matmul(fc2,fc3_weights)+fc3_biases)
        if train:
            fc3=tf.nn.dropout(fc3,dropout_rate)

    #将第三层输出转为RNN输入形式
    fc3=tf.reshape(fc3,[-1,batch_x_shape[0],n_hidden_3])#(n_steps,batch_size,dim)
    #fc3=tf.reshape(fc3,[batch_x_shape[0],-1,n_hidden_3])

    #第四层，双向RNN，输入n_hidden_3，输出2*n_cell_dim
    with tf.variable_scope('layer4-BiLstm',reuse=tf.AUTO_REUSE):
        #前向cell:
        lstm_fw_cell=tf.contrib.rnn.BasicLSTMCell(n_cell_dim,forget_bias=1.0,state_is_tuple=True)
        lstm_fw_cell=tf.contrib.rnn.DropoutWrapper(lstm_fw_cell,input_keep_prob=dropout_rate)
        #反向cell:
        lstm_bw_cell=tf.contrib.rnn.BasicLSTMCell(n_cell_dim,forget_bias=1.0,state_is_tuple=True)
        lstm_bw_cell=tf.contrib.rnn.DropoutWrapper(lstm_bw_cell,input_keep_prob=dropout_rate)
        #time_major == True，则必须是形状的张量： [max_time, batch_size, ...]
        #time_major == False（默认值），则必须是形状的张量： [batch_size, max_time, ...]
        rnn_outputs,output_states=tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,cell_bw=lstm_bw_cell,
                                                              inputs=fc3,dtype=tf.float32,time_major=True,
                                                              sequence_length=seq_length)
        #LSTMCell=keras.layers.LSTMCell(n_cell_dim,unit_forget_bias=True,dropout=dropout_rate)
        #outputs,output_states=keras.layers.Bidirectional(keras.layers.RNN(cell=LSTMCell,return_sequences=True,return_state=True)).input(fc3)
        #keras.layers.Bidirectional(keras.layers.RNN(cell))(input)
        #尺寸为 (batch_size, timesteps, input_dim)

        #连接正、反向结果(n_steps,batch_size,dim)
        rnn_outputs=tf.concat(rnn_outputs,2)
        #转化形状
        rnn_outputs=tf.reshape(rnn_outputs,[-1,2*n_cell_dim])

    #第五层，全连接层，输入2*n_cell_dim，输出n_hidden_5
    with tf.variable_scope('layer5-fc4',reuse=tf.AUTO_REUSE):
        fc4_weights=tf.get_variable(
            "weights",[2*n_cell_dim,n_hidden_5],
            initializer=tf.random_normal_initializer(stddev=0.046875)
        )
        #只有全连接层的权重需要加入正则化
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(fc4_weights))
        fc4_biases=tf.get_variable(
            "biases",[n_hidden_5],initializer=tf.random_normal_initializer(stddev=0.046875)
        )
        fc4=tf.nn.relu(tf.matmul(rnn_outputs,fc4_weights)+fc4_biases)
        if train:
            fc4=tf.nn.dropout(fc4,dropout_rate)

    #第六层，全连接层(输出层)，输入n_hidden_5，输出n_character
    with tf.variable_scope('layer6-fc5',reuse=tf.AUTO_REUSE):
        fc5_weights=tf.get_variable(
            "weights",[n_hidden_5,n_character],
            initializer=tf.random_normal_initializer(stddev=0.046875)
        )
        #只有全连接层的权重需要加入正则化
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(fc5_weights))
        fc5_biases=tf.get_variable(
            "biases",[n_character],initializer=tf.random_normal_initializer(stddev=0.046875)
        )
        fc5=tf.matmul(fc4,fc5_weights)+fc5_biases
        #[n_steps*batch_size,n_character]变为time_major[n_steps,batch_size,n_character]

    fc5=tf.reshape(fc5,[-1,batch_x_shape[0],n_character])
    return fc5

#将所有学习参数定义在CPU的内存，可以让GPU内存充分用于运算
def variable_on_cpu(name,shape,initializer):
    with tf.device('/cpu:0'):
        var=tf.get_variable(name=name,shape=shape,initializer=initializer)
    return var

def train():
    input_tensor = tf.placeholder(tf.float32, [None, None, n_input + (2 * n_input * n_context)],
                                  name='input')
    # ctc_loss计算需要使用sparse_placeholder生成SparseTensor
    targets = tf.sparse_placeholder(tf.int32, name='targets')  # 文本
    keep_dropout = tf.placeholder(tf.float32)
    seq_length = tf.placeholder(tf.int32, [None], name='seq_length')  # 序列长
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    logits=inference(input_tensor,words_size +1,True,keep_dropout,regularizer,tf.to_int64(seq_length))
    avg_loss=tf.reduce_mean(ctc_ops.ctc_loss(targets,logits,seq_length))+tf.add_n(tf.get_collection('losses'))
    learning_rate=0.001
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(avg_loss)
    with tf.name_scope("decode"):
        decoded, log_prob = ctc_ops.ctc_beam_search_decoder(logits, seq_length, merge_repeated=False)
    with tf.name_scope("accuracy"):
        distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), targets)
        # 计算label error rate (accuracy)
        ler = tf.reduce_mean(distance, name='label_error_rate')
    epochs = 100

    saver = tf.train.Saver(max_to_keep=5)  # 生成saver
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        kpt = tf.train.latest_checkpoint(savedir)
        print("kpt:", kpt)
        startepo = 0
        if kpt != None:
            saver.restore(sess, kpt)
            ind = kpt.find("-")
            startepo = int(kpt[ind + 1:])
            print(startepo)

        # 准备运行训练步骤
        section = '\n{0:=^40}\n'
        print(section.format('Run training epoch'))

        train_start = time.time()
        for epoch in range(epochs):  # 样本集迭代次数
            epoch_start = time.time()
            if epoch < startepo:
                continue

            print("epoch start:", epoch+1, "total epochs= ", epochs)
            ##run batch##
            n_batches_per_epoch = int(np.ceil(len(labels) / batch_size))
            print("total loop ", n_batches_per_epoch, "in one epoch，", batch_size, "items in one loop")

            train_cost = 0
            train_ler = 0
            next_idx = 0

            for batch in range(n_batches_per_epoch):  # 一次batch_size，取多少次
                # 取数据
                next_idx, source, source_lengths, sparse_labels =next_batch(labels, next_idx, batch_size)
                feed = {input_tensor: source, targets: sparse_labels, seq_length: source_lengths,
                        keep_dropout: keep_dropout_rate}

                # 计算 avg_loss optimizer ;
                batch_cost, _ = sess.run([avg_loss, optimizer], feed_dict=feed)
                train_cost += batch_cost

                if (batch + 1) % 50 == 0:
                    print('loop:', batch+1, 'Train cost: ', train_cost / (batch + 1))
                    feed2 = {input_tensor: source, targets: sparse_labels, seq_length: source_lengths,
                             keep_dropout: 1.0}

                    d, train_ler = sess.run([decoded[0], ler], feed_dict=feed2)
                    dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=sess)
                    dense_labels = base.sparse_tuple_to_texts_ch(sparse_labels, words)

                    counter = 0
                    print('Label err rate: ', train_ler)
                    duration = time.time() - train_start
                    print('cost time: {:.2f} min'.format(duration / 60))
                    for orig, decoded_arr in zip(dense_labels, dense_decoded):
                        # convert to strings
                        decoded_str = base.ndarray_to_text_ch(decoded_arr, words)
                        print(' file {}'.format(counter))
                        print('Original: {}'.format(orig))
                        print('Decoded:  {}'.format(decoded_str))
                        counter = counter + 1
                        break

            epoch_duration = time.time() - epoch_start

            log = 'Epoch {}/{}, train_cost: {:.3f}, train_ler: {:.3f}, time: {:.2f} sec'
            print(log.format(epoch+1, epochs, train_cost, train_ler, epoch_duration))

            if not os.path.exists(savedir):
                print('不存在训练数据保存目录，现在创建保存目录')
                os.makedirs(savedir)
            saver.save(sess, savedir + "BiRNN.cpkt", global_step=epoch+1)

        train_duration = time.time() - train_start
        print('Training complete, total duration: {:.2f} min'.format(train_duration / 60))

def continue_train():
    input_tensor = tf.placeholder(tf.float32, [None, None, n_input + (2 * n_input * n_context)],
                                  name='input')
    # ctc_loss计算需要使用sparse_placeholder生成SparseTensor
    targets = tf.sparse_placeholder(tf.int32, name='targets')  # 文本
    keep_dropout = tf.placeholder(tf.float32)
    seq_length = tf.placeholder(tf.int32, [None], name='seq_length')  # 序列长
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    logits = inference(input_tensor, words_size + 1, True, keep_dropout, regularizer, tf.to_int64(seq_length))
    avg_loss = tf.reduce_mean(ctc_ops.ctc_loss(targets, logits, seq_length)) + tf.add_n(tf.get_collection('losses'))
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(avg_loss)
    with tf.name_scope("decode"):
        decoded, log_prob = ctc_ops.ctc_beam_search_decoder(logits, seq_length, merge_repeated=False)
    with tf.name_scope("accuracy"):
        distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), targets)
        # 计算label error rate (accuracy)
        ler = tf.reduce_mean(distance, name='label_error_rate')
    epochs = 1000
    #ckpt = tf.train.get_checkpoint_state(savedir)
    saver = tf.train.Saver(max_to_keep=5)
    #saver2 = tf.train.Saver(max_to_keep=5)  # 生成saver
    with tf.Session() as sess:
        choose_cpkt="BiRNN.cpkt-204"
        sess.run(tf.global_variables_initializer())
        print_tensors_in_checkpoint_file(savedir + choose_cpkt,None,True)
        saver.restore(sess, savedir + choose_cpkt)
        #graph = tf.get_default_graph()
        #cur_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        startepo=204
        train_start = time.time()
        for epoch in range(startepo,epochs):  # 样本集迭代次数
            epoch_start = time.time()
            #if epoch < startepo:
            #    continue


            print("epoch start:", epoch+1, "total epochs= ", epochs)
            ##run batch##
            n_batches_per_epoch = int(np.ceil(len(labels) / batch_size))
            print("total loop ", n_batches_per_epoch, "in one epoch，", batch_size, "items in one loop")

            train_cost = 0
            train_ler = 0
            next_idx = 0

            for batch in range(n_batches_per_epoch):  # 一次batch_size，取多少次
                # 取数据
                next_idx, source, source_lengths, sparse_labels =next_batch(labels, next_idx, batch_size)
                feed = {input_tensor: source, targets: sparse_labels, seq_length: source_lengths,
                        keep_dropout: keep_dropout_rate}

                # 计算 avg_loss optimizer ;
                batch_cost, _ = sess.run([avg_loss, optimizer], feed_dict=feed)
                train_cost += batch_cost

                if (batch + 1) % 50 == 0:
                    print('loop:', batch+1, 'Train cost: ', train_cost / (batch + 1))
                    feed2 = {input_tensor: source, targets: sparse_labels, seq_length: source_lengths,
                             keep_dropout: 1.0}

                    d, train_ler = sess.run([decoded[0], ler], feed_dict=feed2)
                    dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=sess)
                    dense_labels = base.sparse_tuple_to_texts_ch(sparse_labels, words)

                    counter = 0
                    print('Label err rate: ', train_ler)
                    duration = time.time() - train_start
                    print('cost time: {:.2f} min'.format(duration / 60))
                    for orig, decoded_arr in zip(dense_labels, dense_decoded):
                        # convert to strings
                        decoded_str = base.ndarray_to_text_ch(decoded_arr, words)
                        decoded_str=decoded_str.strip().strip('龚')
                        print(' file {}'.format(counter))
                        print('Original: {}'.format(orig))
                        print('Decoded:  {}'.format(decoded_str))
                        counter = counter + 1
                        break


            epoch_duration = time.time() - epoch_start

            log = 'Epoch {}/{}, train_cost: {:.3f}, train_ler: {:.3f}, time: {:.2f} sec'
            print(log.format(epoch+1, epochs, train_cost, train_ler, epoch_duration))
            saver.save(sess, savedir + "BiRNN.cpkt", global_step=epoch+1)
            print("save cpkt-%s complete."%(epoch+1))

def test_one_wav(wav_path,label_text):
    tf.reset_default_graph()

    input_tensor = tf.placeholder(tf.float32, [None, None, n_input + (2 * n_input * n_context)],
                                  name='input')
    # ctc_loss计算需要使用sparse_placeholder生成SparseTensor
    targets = tf.sparse_placeholder(tf.int32, name='targets')  # 文本
    keep_dropout = tf.placeholder(tf.float32)
    seq_length = tf.placeholder(tf.int32, [None], name='seq_length')  # 序列长
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    logits = inference(input_tensor, words_size + 1, False, keep_dropout, regularizer, tf.to_int64(seq_length))
    avg_loss = tf.reduce_mean(ctc_ops.ctc_loss(targets, logits, seq_length)) + tf.add_n(tf.get_collection('losses'))
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(avg_loss)
    with tf.name_scope("decode"):
        decoded, log_prob = ctc_ops.ctc_beam_search_decoder(logits, seq_length, merge_repeated=False)
    with tf.name_scope("accuracy"):
        distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), targets)
        # 计算label error rate (accuracy)
        ler = tf.reduce_mean(distance, name='label_error_rate')
    choose_cpkt = "BiRNN.cpkt-117"
    saver = tf.train.Saver(max_to_keep=5)
    re1 = ""
    re2 = ""
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, savedir + choose_cpkt)

        _, source, source_lengths, sparse_labels = next_batch(labels=label_text, wav_files=wav_path)
        feed = {input_tensor: source, targets: sparse_labels, seq_length: source_lengths,
                keep_dropout: 1.0}
        d, test_ler,batch_cost, _ = sess.run([decoded[0], ler,avg_loss, optimizer], feed_dict=feed)
        dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=sess)
        dense_labels = base.sparse_tuple_to_texts_ch(sparse_labels, words)
        print('Label err rate: ', test_ler)
        for orig, decoded_arr in zip(dense_labels, dense_decoded):
            # convert to strings
            decoded_str = base.ndarray_to_text_ch(decoded_arr, words)
            decoded_str = decoded_str.strip().strip('龚')
            re1=orig
            re2=decoded_str
            print('Original: {}'.format(orig))
            print('Decoded:  {}'.format(decoded_str))
    return re1,re2,test_ler





if __name__=='__main__':
    #continue_train()
    test_one_wav("E:/迅雷下载/CSLT public data/thchs30-standalone/wav/train/A2/A2_0.wav")
