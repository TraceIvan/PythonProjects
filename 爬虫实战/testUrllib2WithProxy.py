#用urllib2(urllib.request)使用代理访问网页
import urllib.request,time,platform,os,sys,io,re
def testArgument():
    '''测试输入参数，只需要一个参数'''
    if len(sys.argv)!=2:
        print(u'只需要一个参数,您输入了%s个参数'%(len(sys.argv)))
        tipUse()
        exit()
    else:
        TP=TestProxy(sys.argv[1])

def tipUse():
    '''显示提示信息'''
    print(u'该程序只能输入一个参数，这个参数必须是一个可用的Proxy')
    print(u'usage: python testUrllib2WithProxy.py http://1.2.3.4:5')
    print(u'usage: python testUrllib2WithProxy.py https://1.2.3.4:5')

class TestProxy(object):
    '''这个类的作用是测试Proxy是否有效'''
    def __init__(self,proxy):
        self.proxy=proxy
        self.checkProxyFormat(self.proxy)
        self.url='http://www.baidu.com'
        self.timeout=5
        self.flagWord='百度'#在网页返回的数据中查找这个关键词
        self.useProxy(self.proxy)

    def checkProxyFormat(self,proxy):
        try:
            print(proxy)
            proxyMatch=re.compile('http[s]?://[\d]{1,3}\.[\d]{1,3}\.[\d]{1,3}\.[\d]{1,3}:[\d]{1,5}$')
            re.search(proxyMatch,proxy).group()
        except  AttributeError:
            print('匹配失败')
            tipUse()
            exit()
        flag=1
        proxy=proxy.replace('//','')
        try:
            protocol=proxy.split(':')[0]
            ip=proxy.split(':')[1]
            port=proxy.split(':')[2]
        except IndexError:
            print(u'下标出界')
            tipUse()
            exit()
        flag=flag and len(proxy.split(':'))==3 and len(ip.split('.'))

        flag=ip.split('.')[0] in map(str,range(1,256)) and flag
        flag=ip.split('.')[1] in map(str,range(256)) and flag
        flag = ip.split('.')[2] in map(str, range(256)) and flag
        flag = ip.split('.')[3] in map(str, range(1,256)) and flag

        flag=protocol in [u'http',u'https'] and flag

        flag=port in map(str,range(1,65535)) and flag
        '''这里检查proxy的格式'''
        if flag:
            print(u'输入的http代理服务器符合标准')
        else:
            print(u'输入的http代理服务器不符合标准')
            tipUse()
            exit()

    def useProxy(self,proxy):
        '''利用代理访问百度，并查找关键词'''
        protocol=proxy.split('//')[0].replace(':','')
        ip=proxy.split('//')[1]
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({protocol:ip}))
        urllib.request.install_opener(opener)
        try:
            response=urllib.request.urlopen(self.url,timeout=self.timeout)
        except:
            print(u'连接错误，退出程序')
            exit()
        str=response.read().decode('utf-8')
        if re.search(self.flagWord,str):
            print(u'已取得特征词，该代理可用')
            print(u'获取url信息，response.geturl() :\n%s' % response.geturl())
            print(u'获取返回代码，response.getcode() :\n%s' % response.getcode())
            print(u'获取返回信息，response.info() :\n%s' % response.info())
        else:
            print(u'该代理不可用')

if __name__=='__main__':
    testArgument()
