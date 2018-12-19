import numpy as np
from sklearn import preprocessing

def Mean_removal():
    '''对每一个特征列都缩放到类似的数值范围，每一个特征列的均值为0'''
    data=np.array([[3, -1.5, 2, -5.4],
                   [0, 4,-0.3,2.1],
                   [1, 3.3, -1.9, -4.3]]) # 原始数据矩阵 shape=(3,4)

    data_standardized=preprocessing.scale(data)

    print(data_standardized.shape)
    print('Mean={}'.format(data_standardized.mean(axis=0)))
    print('Mean2={}'.format(np.mean(data_standardized,axis=0)))
    print('standardized: ')
    print(data_standardized)
    print('STD={}'.format(np.std(data_standardized,axis=0)))

def Scaling():
    '''将特征列的数值范围缩放到合理的大小'''
    data = np.array([[3, -1.5, 2, -5.4],
                     [0, 4, -0.3, 2.1],
                     [1, 3.3, -1.9, -4.3]])  # 原始数据矩阵 shape=(3,4)

    data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # 缩放到（0,1）之间
    data_scaled = data_scaler.fit_transform(data)

    print('scaled matrix: *********************************')
    print(data_scaled)

def Normalization():
    '''归一化，将特征向量调整为L1范数或L2范数，使特征向量的数值之和为1'''
    data = np.array([[3, -1.5, 2, -5.4],
                     [0, 4, -0.3, 2.1],
                     [1, 3.3, -1.9, -4.3]])  # 原始数据矩阵 shape=(3,4)

    data_L1_normalized = preprocessing.normalize(data, norm='l1')
    print('L1 normalized matrix: *********************************')
    print(data_L1_normalized)
    print('sum of matrix: {}'.format(np.sum(data_L1_normalized)))

    data_L2_normalized = preprocessing.normalize(data)  # 默认：l2
    print('L2 normalized matrix: *********************************')
    print(data_L2_normalized)
    print('sum of matrix: {}'.format(np.sum(data_L2_normalized)))

def Binarization():
    '''二值化，将数值特征向量转换为布尔类型向量
    二值化之后的数据点都是0或者1,将所有大于threshold的数据都改为1，小于等于threshold的都设为0
    '''
    data = np.array([[3, -1.5, 2, -5.4],
                     [0, 4, -0.3, 2.1],
                     [1, 3.3, -1.9, -4.3]])  # 原始数据矩阵 shape=(3,4)
    data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
    print('binarized matrix: *********************************')
    print(data_binarized)

def One_Hot_Encoding():
    '''独热编码，当数值稀疏时，用来缩小特征向量的维度'''
    data = np.array([[0, 2, 1, 12],
                     [1, 3, 5, 3],
                     [2, 3, 2, 12],
                     [1, 2, 4, 3]])  # 原始数据矩阵 shape=(4,4)
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(data)
    encoded_vector = encoder.transform([[2, 3, 5, 3]]).toarray()
    print('one-hot encoded matrix: *********************************')
    print(encoded_vector.shape)#shape=(1,11)
    print(encoded_vector)
    '''
编码方式为：根据原始数据集data构建编码器encoder，用编码器来对新数据进行编码。比如，第0列有三个不同值（0,1,2），故而有三个维度，
即0=100，1=010，2=001；同理，第1列有两个不同值（2,3），故而只有两个维度，即2=10，3=01；同理，第2列有四个不同值（1,5,2,4），
故而有四个维度，即1=1000，2=0100,4=0010,5=0001同理，第3列有两个不同值（3,12），故而只有两个维度，即3=10，12=01。
所以在面对新数据[[2,3,5,3]]时，第0列的2就对应于001，第二列的3对应于01，第三列的5对应于0001，第四列的3对应于10，
连接起来后就是输出的这个（1,11）矩阵，即为读了编码后的致密矩阵。
如果面对的新数据不存在上面的编码器中，比如[[2,3,5,4]]时，4不存在于第3列（只有两个离散值3和12），则输出为00，
连接起来后是[[0. 0. 1. 0. 1. 0. 0. 0. 1. 0. 0.]]，注意倒数第二个数字变成了0
    '''


def label_encoding():
    '''对标记(类别值等)进行编码'''
    # 构建编码器
    encoder = preprocessing.LabelEncoder()  # 先定义一个编码器对象
    raw_labels = ['翠花', '张三', '王宝强', '芙蓉姐姐', '凤姐', '王宝强', '凤姐']
    encoder.fit(raw_labels)  # 返回自己的一个实例
    print('编码器列表：{}'.format(encoder.classes_))  # 返回编码器中所有类别，已经排除了重复项
    for index, item in enumerate(encoder.classes_):
        print('{} --> {}'.format(item, index))#凤姐 ->0，张三 ->1，王宝强 ->2，翠花 ->3，芙蓉姐姐->4

    # 使用编码器来编码新样本数据
    need_encode_labels = ['王宝强', '芙蓉姐姐', '翠花']
    # need_encode_labels=['王宝强','芙蓉姐姐','翠花','无名氏']
    # 在编码时，如果遇到编码器中没有的标记时会报错，在解码时也一样，如'无名氏'
    encoded_labels = encoder.transform(need_encode_labels)
    print('\n编码之前的标记：{}'.format(need_encode_labels))
    print('编码之后的标记：{}'.format(encoded_labels))

    # 使用编码器将编码数字解码成原来的文本标记，注意最大值不能超过编码器中的长度
    encoded = [1, 3, 0, 4]
    # encoded=[1,3,0,4,5] # 5不存在与编码器中，故报错
    decoded_labels = encoder.inverse_transform(encoded)
    print('\n已经编码的标记代码：{}'.format(encoded))
    print('解码后的标记：{}'.format(decoded_labels))

def evaluation_index(predict_data,true_ret):
    '''使用评价指标来评估模型的好坏'''
    import sklearn.metrics as metrics
    print('平均绝对误差：{}'.format(
        round(metrics.mean_absolute_error(predict_data, true_ret), 2)))
    print('均方误差MSE：{}'.format(
        round(metrics.mean_squared_error(predict_data, true_ret), 2)))
    print('中位数绝对误差：{}'.format(
        round(metrics.median_absolute_error(predict_data, true_ret), 2)))
    print('解释方差分：{}'.format(
        round(metrics.explained_variance_score(predict_data, true_ret), 2)))
    print('R方得分：{}'.format(
        round(metrics.r2_score(predict_data, true_ret), 2)))

def cal_H(x):
    '''计算信息熵H(x)'''
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent

def cal_condition_H(x, y):
    '''计算条件信息熵H(y|x)'''
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        temp_ent = cal_H(sub_y)
        ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent
    return ent

def cal_ent_grap(x,y):
    '''计算信息增益'''
    base_ent = cal_H(y)
    condition_ent = cal_condition_H(x, y)
    ent_grap = base_ent - condition_ent
    return ent_grap

def cal_MI(labels_true,labels_pred):
    '''计算互信息、标准互信息'''
    from sklearn.metrics import mutual_info_score,normalized_mutual_info_score
    MI=mutual_info_score(labels_true,labels_pred)
    NMI=normalized_mutual_info_score(labels_true,labels_pred)
    return MI,NMI