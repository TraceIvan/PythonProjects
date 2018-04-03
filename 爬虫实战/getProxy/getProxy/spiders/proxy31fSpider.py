# -*- coding: utf-8 -*-
import scrapy
from getProxy.items import GetproxyItem

class Proxy31fspiderSpider(scrapy.Spider):
    name = 'proxy31fSpider'
    allowed_domains = ['31f.cn']
    start_urls = ['http://31f.cn/']

    def parse(self, response):
        subselector=response.xpath('//tr')
        items=[]
        for sub in subselector[1:]:
            item=GetproxyItem()
            item['ip']=sub.xpath('./td[2]//text()').extract()[0]
            item['port']=sub.xpath('./td[3]//text()').extract()[0]
            item['type']=""
            item['loction']=sub.xpath('./td[4]//text()').extract()[0]
            item['protocol']=sub.xpath('./td[5]//text()').extract()[0]
            item['exitdays']=sub.xpath('./td[7]//text()').extract()[0]
            item['source']='http://31f.cn/'
            items.append(item)
        return items
