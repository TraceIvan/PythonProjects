from pandas import read_csv
from datetime import datetime
from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def pre_process():
    # load data
    def parse(x):
        return datetime.strptime(x, '%Y %m %d %H')
    path='D:/数据集/气候/beijing_PM2.5_data.csv'
    dataset = read_csv(path, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse,engine='python')
    print(dataset)
    dataset.drop('No', axis=1, inplace=True)#丢弃NO列
    # manually specify column names
    #PM2.5浓度、Dew Point、温度、压力、风向、风速、累计小时雪、累计小时雨
    dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'
    # mark all NA values with 0
    dataset['pollution'].fillna(0, inplace=True)#将NA值替换为“0”值
    # drop the first 24 hours
    dataset = dataset[24:]
    # summarize first 5 rows
    print(dataset.head(5))
    # save to file
    dataset.to_csv('pollution.csv')

def visual_data():
    from pandas import read_csv
    from matplotlib import pyplot
    # load dataset
    dataset = read_csv('pollution.csv', header=0, index_col=0)
    values = dataset.values
    # specify columns to plot
    groups = [0, 1, 2, 3, 5, 6, 7]
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(dataset.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.show()

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def prepare():
    # load dataset
    dataset = read_csv('pollution.csv', header=0, index_col=0)
    values = dataset.values
    # integer encode direction
    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])#将非数值型风向编码
    # ensure all data is float
    values = values.astype('float32')#转化为float型
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # specify the number of lag hours
    n_hours = 3
    n_features = 8
    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_hours, 1)

    # specify the number of lag hours
    n_hours = 1
    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_hours, 1)
    print(reframed)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    print(reframed.head())
    return reframed,scaler

def split(reframed):
    # split into train and test sets
    values = reframed.values
    n_train_hours = 365 * 24
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    print(train_X.shape)#(8760, 8)
    print(train_X.shape[1])#8
    print(test_X.shape)#(35039, 8)
    print(test_X.shape[1])#8
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    return train_X,train_y,test_X,test_y

def LSTM_process(train_X,train_y,test_X,test_y):
    # design network
    model = Sequential()
    #在第一个隐层中定义具有50 个神经元的LSTM ,输入形状是1 个时间步长，具有8 个特征。
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    #用于预测污染的输出层中的1个神经元
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    return model,history

def predict(model,test_X,test_y,scaler):
    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

if __name__=='__main__':
    #pre_process()
    visual_data()
    reframed,scaler=prepare()
    train_X, train_y, test_X, test_y=split(reframed)
    model,history=LSTM_process(train_X, train_y, test_X, test_y)
    predict(model,test_X, test_y,scaler)

