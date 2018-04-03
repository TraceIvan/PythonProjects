# -*- coding: utf-8 -*-


import yqd

def load_quote(ticker):
	print('===', ticker, '===')
	print(yqd.load_yahoo_quote(ticker, '20170515', '20170517'))
	print(yqd.load_yahoo_quote(ticker, '20170515', '20170517', 'dividend'))
	print(yqd.load_yahoo_quote(ticker, '20170515', '20170517', 'split'))

def test():
	# Download quote for stocks
	load_quote('QCOM')
	load_quote('C')

	# Download quote for index
	load_quote('^DJI')

if __name__ == '__main__':
	test()