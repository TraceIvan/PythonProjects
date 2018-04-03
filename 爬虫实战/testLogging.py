import logging

class TestLogging(object):
    def __init__(self):
        logFormat='%(asctime)-12s %(levelname)-8s %(name)-10s %(message)-12s'#当前时间、日志级别、Logger名字、用户输出的信息
        logFileName='./testLog.txt'

        logging.basicConfig(level=logging.INFO,
                            format=logFormat,
                            filename=logFileName,
                            filemode='w')

        logging.debug('debug message')
        logging.info('info message')
        logging.warning('warning message')
        logging.error('error message')
        logging.critical('critical message')

if __name__=='__main__':
    tl=TestLogging()