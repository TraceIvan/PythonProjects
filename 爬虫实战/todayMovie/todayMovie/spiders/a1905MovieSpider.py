# -*- coding: utf-8 -*-
import scrapy
from todayMovie.items import TodaymovieItem

class A1905moviespiderSpider(scrapy.Spider):
    name = '1905MovieSpider'
    allowed_domains = ['vip.1905.com']
    start_urls = ['http://vip.1905.com/list/p1o6.shtml']

    def parse(self, response):
        subSelector=response.xpath('//span[@class="name"]')#嵌套选择

        items=[]
        for sub in subSelector:
            item=TodaymovieItem()
            item['movieName']=sub.xpath('text()').extract()[0]
            items.append(item)
        return items
