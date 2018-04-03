from scrapy.contrib.downloadermiddleware.useragent import UserAgentMiddleware
from meijunew.MiddleWares.resource import UserAgents
import random

class RandomUserAgent(UserAgentMiddleware):
    def process_request(self, request, spider):
        ua=random.choice(UserAgents)
        request.headers.setdefault('User-Agent',ua)
