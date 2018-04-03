import urllib.request,userAgents
'''userAgents是个自定义的模块，位置处于当前目录下'''
class UrllibModifyHeader(object):
    '''使用Urllib2(urllib.response)修改header'''
    def __init__(self):
        #这个是PC+IE的User-Agent
        PIUA=userAgents.pcUserAgent.get('IE 9.0')
        #这个是Mobile+UC的User-Agent
        MUUA=userAgents.mobileUserAgent.get('UC standard')
        #测试网站选择有道翻译
        self.url='http://fanyi.youdao.com'

        self.useUserAgent(PIUA,1)
        self.useUserAgent(MUUA,2)

    def useUserAgent(self,userAgnet,name):
        request=urllib.request.Request(self.url)

        request.add_header(userAgnet.split(':')[0],userAgnet.split(':')[1])
        response=urllib.request.urlopen(request)
        fileName=str(name)+'.html'
        with open(fileName,'a',encoding='utf-8') as fp:
            fp.write("%s\n\n"%userAgnet)
            fp.write(response.read().decode('utf-8'))


if __name__=='__main__':
    umh=UrllibModifyHeader()