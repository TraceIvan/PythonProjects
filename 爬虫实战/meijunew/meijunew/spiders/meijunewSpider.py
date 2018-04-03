# -*- coding: utf-8 -*-
import scrapy
from meijunew.items import MeijunewItem

class MeijunewspiderSpider(scrapy.Spider):
    name = 'meijunewSpider'
    allowed_domains = ['meijutt.com']
    start_urls = ['http://www.meijutt.com/new100.html']

    def parse(self, response):
        subSelector=response.xpath('//div[@class="wrap"]//li')

        items=[]
        for sub in subSelector:
            item=MeijunewItem()
            item['storyName']=sub.xpath('./h5//text()').extract()[0].strip()
            item['storyState']=sub.xpath('./span[@class="state1 new100state1"]/font//text()').extract()
            if item['storyState']:item['storyState']=item['storyState'][0].strip()
            else:item['storyState']=sub.xpath('./span[@class="state1 new100state1"]//text()').extract()[0].strip()
            item['storyType']=sub.xpath('./span[@class="mjjq"]/text()').extract()[0].strip()
            item['tvStation']=sub.xpath('./span[@class="mjtv"]/text()').extract()[0].strip()
            item['updateTime']=sub.xpath('./div[@class="lasted-time new100time fn-right"]/font//text()').extract()
            if item['updateTime']:item['updateTime']=item['updateTime'][0].strip()
            else:item['updateTime']=sub.xpath('./div[@class="lasted-time new100time fn-right"]//text()').extract()[0].strip()
            items.append(item)
        return items
