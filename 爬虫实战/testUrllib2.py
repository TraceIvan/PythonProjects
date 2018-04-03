import urllib.request,time,platform,os,sys,io
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8') #改变标准输出的默认编码
def clear():
    print(u'内容较多，显示3秒后翻页')
    time.sleep(3)
    OS=platform.system()
    if OS==u'windows':
        os.system('cls')
    else:
        os.system('clear')

def linkBaidu():
    url='http://www.baidu.com'
    response=''
    try:
        response=urllib.request.urlopen(url,timeout=3)
    except urllib.request.URLError:
        print(u'网络地址错误')
        exit()
    with open('./baidu.txt','w',encoding='utf-8') as fp:#当文本文件里面有中文时，需要进行编码转换
        fp.write(response.read().decode('utf-8'))
    print(u'获取url信息，response.geturl() :\n%s'%response.geturl())
    print(u'获取返回代码，response.getcode() :\n%s' % response.getcode())
    print(u'获取返回信息，response.info() :\n%s' % response.info())
    print(u'获取的网页信息已存入当前目录的baidu.txt中，请自行查看')

if __name__=='__main__':
    linkBaidu()