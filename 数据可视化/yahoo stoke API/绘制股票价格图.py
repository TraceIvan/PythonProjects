from pylab import plotfile,show,gca
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas_datareader.data as web
import numpy as np
import datetime

import blockspring
import json

def methodFirst():

    fname=cbook.get_sample_data(r'D:\Python\数据可视化\yahoo stoke API\amzn_2014-1-1_today.csv',asfileobj=False)
    plotfile(fname,('date','high','low','close'),subplots=False)
    show()
    plotfile(fname,(0,1,5),plotfuncs={5:'bar'})
    show()

def methodSecond():
    '''显示baidu、google、Amazon从2014-1-1至今的收盘价'''
    plt.rc('axes', grid=True)
    plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)
    fig = plt.figure(facecolor='white')
    axescolor = '#f6f6f6'
    ax = fig.add_subplot(111, facecolor=axescolor)
    startDate = datetime.date(2014, 1, 1)
    today = endDate = datetime.date.today()
    ticker = ['BIDU', 'GOOG', 'AMZN']

    plotTicker(ticker[0], startDate, endDate, 'red', ax)
    plotTicker(ticker[1], startDate, endDate, 'blue', ax)
    plotTicker(ticker[2], startDate, endDate, 'green', ax)
    plt.show()

def plotTicker(ticker,startDate,endDate,fillColor,ax):
    pd_DataFrame = web.DataReader(ticker, 'iex', startDate, endDate)
    dates=pd_DataFrame.index.tolist()
    dates=[datetime.datetime.strptime(d,'%Y-%m-%d').date() for d in dates]
    closes=pd_DataFrame['close'].tolist()
    map(float,closes)
    ax.plot(dates,closes,color=fillColor,lw=2,label=ticker)
    ax.legend(loc='upper right',shadow=True,fancybox=True)

    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment('right')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())

def methodThird():
    '''同时绘制微软的每天收盘价和总成交额'''
    plt.rc('axes', grid=True)
    plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)
    fig = plt.figure(facecolor='white')
    axescolor = '#f6f6f6'
    ax = fig.add_subplot(111, facecolor=axescolor)
    startDate = datetime.date(2014, 1, 1)
    today = endDate = datetime.date.today()
    ticker = 'MSFT'#微软公司
    plotSingleTickerWithVolume(ticker,startDate,endDate,ax)
    plt.show()

def plotSingleTickerWithVolume(ticker,startDate,endDate,ax):
    pd_DataFrame = web.DataReader(ticker, 'iex', startDate, endDate)
    dates=pd_DataFrame.index.tolist()
    dates=[datetime.datetime.strptime(d,'%Y-%m-%d').date() for d in dates]
    dates=np.array(dates)
    closes=pd_DataFrame['close'].tolist()
    map(float,closes)
    closes=np.array(closes)
    volumes=pd_DataFrame['volume'].tolist()
    map(float,volumes)
    volumes=np.array(volumes)
    axt=ax.twinx()
    fcolor='darkgoldenrod'

    ax.plot(dates,closes,color='#1066ee',lw=2,label=ticker)
    ax.fill_between(dates,closes,0,facecolors='#BBD7E5')
    ax.legend(loc='upper right',shadow=True,fancybox=True)
    total_volumes=closes*volumes/1e6#单位：1,000,000美元
    vmax=total_volumes.max()
    axt.plot(dates, total_volumes, color=fcolor, lw=2, label=ticker)
    axt.fill_between(dates,total_volumes,0,label='Tot_volume',facecolors=fcolor,
                     edgecolor=fcolor)
    #axt.set_yticks([])

    for axis in ax,axt:
        for label in axis.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment('right')
        axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axis.xaxis.set_major_locator(mdates.MonthLocator())

def methodForth():
    print(blockspring.runParsed("stock-price-comparison",
                                {
                                    "tickers":"FB,BIDU,TWTR",
                                    "start_date":"2014-01-01",
                                    "end_date":"2015-01-01"}))

if __name__=='__main__':
    #methodFirst()
    #methodSecond()
    #methodThird()
    methodForth()
