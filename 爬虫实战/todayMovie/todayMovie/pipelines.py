# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

import time

class TodaymoviePipeline(object):
    def process_item(self, item, spider):
        now=time.strftime('%Y-%m-%d',time.localtime())
        filename='vip1905'+now+'.txt'
        with open(filename,'a') as fp:
            fp.write(item['movieName']+'\n\n')
        return item
