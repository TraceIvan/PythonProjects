# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import time
import os.path
import csv

class WeatherPipeline(object):
    def process_item(self, item, spider):
        today=time.strftime('%Y%m%d',time.localtime())
        filename=today+'.csv'
        fieldname= list(dict(item).keys())
        if not os.path.isfile(filename):
            with open(filename, 'a', encoding='GB2312', newline='') as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldname)
                writer.writeheader()
        with open(filename,'a',encoding='GB2312',newline='') as fp:
            writer=csv.DictWriter(fp,fieldnames=fieldname)
            writer.writerow(dict(item))
        return item
