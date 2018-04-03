# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class GetproxyItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    ip=scrapy.Field()#服务器IP
    port=scrapy.Field()#服务器端口
    type=scrapy.Field()#代理类型（高匿等）
    loction=scrapy.Field()#（地区）
    protocol=scrapy.Field()#协议
    exitdays=scrapy.Field()#存活时间
    source=scrapy.Field()#来源网址
    pass
