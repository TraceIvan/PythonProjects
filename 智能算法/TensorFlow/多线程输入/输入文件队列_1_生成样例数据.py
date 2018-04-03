import tensorflow as tf

#创建TFRecord文件帮助函数
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#模拟海量数据情况下将数据写入不同的文件。
#num_shards定义总共写入多少个文件，instance_per_shard定义每个文件有多少个数据
num_shards=2
instances_per_shard=2
for i in range(num_shards):
    #将数据分为多个文件时，可以将不同文件以类似0000n-of-0000m的后缀区分
    #表示当前文件编号和总文件数目
    filename=('data.tfrecords-%.5d-of-%.5d'%(i,num_shards))
    writer=tf.python_io.TFRecordWriter(filename)
    #将数据封装成Example结构并写入TFRecord文件
    for j in range(instances_per_shard):
        #此处Example仅包含当前样例为第几个文件的第几个样本数据
        example=tf.train.Example(features=tf.train.Features(feature={
            'i':_int64_feature(i),
            'j':_int64_feature(j)
        }))
        writer.write(example.SerializeToString())
    writer.close()
