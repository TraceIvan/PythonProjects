# -*- coding: utf-8 -*-
import scrapy
from qiushi.items import QiushiItem

class QiushispiderSpider(scrapy.Spider):
    name = 'qiushiSpider'
    allowed_domains = ['qiushibaike.com']
    pages=13
    start_urls = []
    for i in range(1,pages+1):
        start_urls.append('http://www.qiushibaike.com/page/'+str(i))

    def parse(self, response):
        subselector=response.xpath('//div[contains(@class,"article block untagged mb15")]')
        items=[]
        for sub in subselector:
            item=QiushiItem()
            item['author']=sub.xpath('./div[contains(@class,"author")]//h2/text()').extract()[0].strip()
            content=sub.xpath('./a[contains(@href,"article")]//span//text()').extract()
            item['content']=""
            if content:
                for sb in content:
                    item['content']=item['content']+sb.strip().replace('\n\t','')+'\n'
            item['funNum']=sub.xpath('./div[@class="stats"]/span[@class="stats-vote"]//i/text()').extract()[0].strip()
            item['talkNum']=sub.xpath('./div[@class="stats"]/span[@class="stats-comments"]//i/text()').extract()[0].strip()
            if sub.xpath('./div[@class="thumb"]//img/@src').extract():
                item['img']=sub.xpath('./div[@class="thumb"]//img/@src').extract()[0].strip()
            items.append(item)
        return items

