import copy
import numpy as np

np.random.seed(0)
def sigmoid(x):
    output=1/(1+np.exp(-x))
    return output

def sigmoid_output_to_derivative(output):#sigmoid函数导数
    return output*(1-output)

int2binary={}#整数到二进制映射
binary_dim=8#8位二进制内
#计算0~255的二进制表示
largest_number=pow(2,binary_dim)
binary=np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i]=binary[i]

#参数设置
alpha=0.9 #学习率
input_dim=2 #输入维度
hidden_dim=16
output_dim=1#输入维度

#初始化网络权重（限制在[-0.05,0.05)范围）
synapse_0=(2*np.random.random((input_dim,hidden_dim))-1)*0.05#输入层，输入2*16
synapse_1=(2*np.random.random((hidden_dim,output_dim))-1)*0.05#输出层
synapse_h=(2*np.random.random((hidden_dim,hidden_dim))-1)*0.05#循环节点
#存放反向传播的权重更新值
synapse_0_update=np.zeros_like(synapse_0)
synapse_1_update=np.zeros_like(synapse_1)
synapse_h_update=np.zeros_like(synapse_h)





#开始训练
for j in range(10000):
    #生成被减数和减数和实际值
    a_int=np.random.randint(largest_number)
    b_int=np.random.randint(largest_number/2)
    if a_int<b_int:
        a_int,b_int=b_int,a_int

    a=int2binary[a_int]
    b=int2binary[b_int]
    c_int=a_int-b_int
    c=int2binary[c_int]

    #存储预测值
    d=np.zeros_like(c)

    overAllError = 0  # 初始化总误差
    layer_2_deltas = list()  # 存储每个时间点输出层的误差
    layer_1_values = list()  # 存储每个时间点隐藏层的值
    layer_1_values.append(np.ones(hidden_dim) * 0.1)  # 设置初始值为0.1

    #正向传播
    for position in range(binary_dim):#循环遍历每一个二进制位
        X=np.array([[a[binary_dim-position-1],b[binary_dim-position-1]]])#从右到左
        y=np.array([[c[binary_dim-position-1]]]).T #正确答案
        layer_1=sigmoid(np.dot(X,synapse_0)+np.dot(layer_1_values[-1],synapse_h))#当前隐藏层输出=输入层+之前的隐藏层输出
        layer_2=sigmoid(np.dot(layer_1,synapse_1))
        layer_2_error=y-layer_2#预测误差
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))#每个时间点的导数
        overAllError+=np.abs(layer_2_error[0])
        d[binary_dim-position-1]=np.around(layer_2[0][0])#记录每一个预测bit位
        layer_1_values.append(copy.deepcopy(layer_1))#记录新的隐藏层输出

    #反向传播
    future_layer_1_delta=np.zeros(hidden_dim)
    for position in range(binary_dim):

        X=np.array([[a[position],b[position]]])#最后一次的两个输入
        cur_layer_1=layer_1_values[-position-1]#当前时间点的隐藏层
        prev_layer_1=layer_1_values[-position-2]#前一个时间点的隐藏点

        cur_layer_2_deltas=layer_2_deltas[-position-1]#当前时间点输出层导数
        #通过后一个时间点的隐藏层误差和当前时间点的输出层误差，计算当前时间点的隐藏层误差
        layer_1_delta=(future_layer_1_delta.dot(synapse_h.T)+cur_layer_2_deltas.dot(synapse_1.T))*sigmoid_output_to_derivative(cur_layer_1)
        #完成所有反向传播误差计算后，更新权重矩阵。此处先存储
        synapse_1_update+=np.atleast_2d(cur_layer_1).T.dot(cur_layer_2_deltas)
        synapse_h_update+=np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update+=X.T.dot(layer_1_delta)
        future_layer_1_delta=layer_1_delta
    # 完成所有反向传播误差计算后，更新权重矩阵
    synapse_0+=synapse_0_update*alpha
    synapse_1+=synapse_1_update*alpha
    synapse_h+=synapse_h_update*alpha
    synapse_0_update*=0
    synapse_1_update*=0
    synapse_h_update*=0

    if (j+1)%500==0:
        print("总误差："+str(overAllError))
        print("Pred:"+str(d))
        print("True:"+str(c))
        out=0
        for index,x in enumerate(reversed(d)):
            out+=x*pow(2,index)
        print(str(a_int)+"-"+str(b_int)+"="+str(out))
        print("-------------")

