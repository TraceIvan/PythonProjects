# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

import time

class MeijunewPipeline(object):
    def process_item(self, item, spider):
        today=time.strftime('%Y-%m-%d',time.localtime())
        filename=today+'-meijunew.txt'
        with open(filename,'a') as fp:
            fp.write('*'*100+'\n')
            fp.write('storyName:%s\n'%(item['storyName']))
            fp.write('storyState:%s\n'%(item['storyState']))
            fp.write('storyType:%s\n'%(item['storyType']))
            fp.write('tvStation:%s\n'%(item['tvStation']))
            fp.write('updateTime:%s\n'%(item['updateTime']))
            fp.write('*'*100+'\n\n')

        return item
