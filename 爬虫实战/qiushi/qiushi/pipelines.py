# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

import time
import urllib.request as urllib2
import os

class QiushiPipeline(object):
    def process_item(self, item, spider):
        today=time.strftime('%Y%m%d',time.localtime())
        filename=today+'_qiushi.txt'
        imgDir='IMG'
        if os.path.isdir(imgDir):
            pass
        else:
            os.makedirs(imgDir)
        with open(filename,'a') as fp:
            fp.write('-'*50+'\n'+'*'*50+'\n')
            fp.write("author:\t %s\n"%(item['author']))
            fp.write("content:\t %s\n"%(item['content']))
            try:
                imgUrl=item['img']
            except IndexError and KeyError:
                pass
            else:
                imgName=os.path.basename(imgUrl)
                imgDir=os.path.abspath(imgDir)
                fp.write("img:\t %s\n"%(imgName))
                imgPathName=imgDir+os.sep+imgName
                with open(imgPathName,'wb') as fpi:
                    response=urllib2.urlopen("http:"+imgUrl)
                    fpi.write(response.read())
            fp.write("fun:%s\t talk:%s\n"%(item['funNum'],item['talkNum']))
            fp.write('*'*50+'\n'+'-'*50+'\n'*5)
            #time.sleep(1)
        return item
