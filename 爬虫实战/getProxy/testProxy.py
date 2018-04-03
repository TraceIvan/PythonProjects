import urllib.request as urllib2
import re
import threading
import time

class TestProxy(object):
    def __init__(self):
        now = time.strftime('%Y-%m-%d', time.localtime())
        filename=now+'_proxy.txt'
        self.sFile=filename
        self.dFile=now+'_alive.txt'
        self.URL='http://vchart.yinyuetai.com/vchart/trends?area=JP'
        self.threads=10
        self.timeout=3
        self.regex=re.compile('yinyuetai')
        self.aliveList=[]

        self.run()

    def run(self):
        with open(self.sFile,'r',encoding='utf8') as fp:
            lines=fp.readlines()
            line=lines.pop()
            while lines:
                for i in range(self.threads):
                    t=threading.Thread(target=self.linkWithProxy,args=(line,))#多线程并发

                    t.start()
                    if lines:
                        line=lines.pop()
                    else:
                        continue
        with open(self.dFile, 'w',encoding='utf8') as fp:
            for i in range(len(self.aliveList)):
                fp.write(self.aliveList[i])
        '''
        ss=set()
        with open(self.dFile,'w') as fp:
            for i in range(len(self.aliveList)):
                if(self.aliveList[i].split('\t')[0] not in ss):
                    fp.write(self.aliveList[i])
                    ss.add(self.aliveList[i].split('\t')[0])
        '''

    def linkWithProxy(self,line):
        lineList=line.split('\t')
        protocol=lineList[2].strip().lower()
        server=protocol+'://'+lineList[0].strip()+':'+lineList[1].strip()
        opener=urllib2.build_opener(urllib2.ProxyHandler({protocol:server}))
        urllib2.install_opener(opener)
        try:
            response=urllib2.urlopen(self.URL,timeout=self.timeout)
        except:
            print("%s connect failed"%server)
            return
        else:
            try:
                str=response.read()
            except:
                print('%s connect failed'%server)
                return
            if self.regex.search(str.decode('utf8')):
                print('%s connect success ......'%server)
                self.aliveList.append(line)

if __name__=='__main__':
    TP=TestProxy()