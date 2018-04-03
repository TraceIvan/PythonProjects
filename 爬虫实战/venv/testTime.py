import time
from myLog  import MyLog

class TestTime(object):
    def __init__(self):
        self.log=MyLog()
        self.testTime()
        self.testLocaltime()
        self.testSleep()
        self.testStrftime()
    def testTime(self):
        self.log.info(u'开始测试time.time()函数')
        print(u'当前时间戳为：time.time()=%f'%time.time())
        print(u'这里返回的是一个浮点型的数值，它是从1970纪元后经过的浮点秒数')
        print('\n')

    def testLocaltime(self):
        self.log.info(u'开始测试time.localtime()函数')
        print(u'当前本地时间为：time.localtime()= %s'%str(time.localtime()))
        print(u'这里返回的是一个struct_time结构的元组')
        print('\n')

    def testSleep(self):
        self.log.info(u'开始测试time.sleep()函数')
        print(u'这是个计时器：time.sleep(5)')
        print(u'闭上眼睛数上5s就可以')
        time.sleep(5)
        print('\n')

    def testStrftime(self):
        self.log.info(u'开始测试time.strftime()函数')
        print(u'这个函数返回的是一个格式化的时间')
        print(u'time.strftime("%%Y-%%m-%%d %%X",time.localtime())= %s'%time.strftime("%Y-%m-%d %X",time.localtime()))
        print('\n')


if __name__=='__main__':
    print(time.time())
    print(time.localtime())
    for i in range(5):
        time.sleep(1)
        print(i)
        print(time.strftime('%Y-%m-%d %X %a',time.localtime()))

    tt=TestTime()