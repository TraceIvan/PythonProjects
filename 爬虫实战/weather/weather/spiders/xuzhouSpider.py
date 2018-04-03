# -*- coding: utf-8 -*-
import scrapy
from weather.items import WeatherItem

class XuzhouspiderSpider(scrapy.Spider):
    name = 'xuzhouSpider'
    allowed_domains = ['tianqi.so.com']
    start_urls = ['http://tianqi.so.com/weather/101190801']
    city=['xuzhou']

    def parse(self, response):
        subSelector=response.xpath('//ul[@class="weather-columns"]')
        items=[]
        for sub in subSelector:
            item=WeatherItem()
            item['cityDate']=sub.xpath('./li/div[2]/text()').extract()[0].strip()
            item['week']=''
            item['img']=''
            item['weather']=sub.xpath('./li/div[4]/text()').extract()[0].strip()
            item['temperature']=sub.xpath('./li/div[5]/text()').extract()[0].strip()
            if sub.xpath('.//div[contains(@class,"aqi-label")]'):
                item['airquality']=sub.xpath('./li/div[6]/text()').extract()[0].strip()
                item['wind'] = sub.xpath('./li/div[7]/text()').extract()[0].strip()
            else:
                item['airquality'] = ''
                item['wind'] = sub.xpath('./li/div[6]/text()').extract()[0].strip()
            items.append(item)
        return items

