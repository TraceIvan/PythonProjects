# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import time
import os.path
import urllib.request

class WeatherPipeline(object):
    def process_item(self, item, spider):
        today=time.strftime('%Y%m%d',time.localtime())
        filename=today+'.txt'
        with open(filename,'a') as fp:
            fp.write("{:<15}".format(item['cityDate'])+'\t')
            #fp.write(item['week']+'\t')
            '''
            imgName=os.path.basename(item['img'])
            fp.write(imgName+'\t')
            if  item['img'] and os.path.exists(imgName):
                pass
            else:
                with open(imgName,'wb') as fp2:
                    response=urllib.request.urlopen(item['img'])
                    fp.write(response.read())
            '''
            fp.write("{:<10}".format(item['weather'])+'\t')
            fp.write("{:>8}".format(item['temperature'])+'\t')
            fp.write("{:<8}".format(item['airquality'])+'\t')
            fp.write("{:<15}".format(item['wind'])+'\t')
            fp.write('\n\n')
            time.sleep(1)
        return item
