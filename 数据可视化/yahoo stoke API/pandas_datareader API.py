import datetime
from matplotlib import mlab
import pandas as pd
import pandas_datareader.data as web
import pandas_datareader as pdr


if __name__=='__main__':
    startDate=datetime.date(2014,1,1)
    today=endDate=datetime.date.today()
    ticker='SINA'
    '''
    TWTR:Twitter
    BIDU:Baidu
    GOOG:Google
    AMZN:Amazon
    NTES:网易
    SINA:新浪
    FB:Facebook
    '''
    fh=web.DataReader(ticker,'iex',startDate,endDate)
    fh.to_csv(ticker.lower()+'_2014-1-1_today.csv')
    #开盘价、最高价、最低价、收盘价、交易量