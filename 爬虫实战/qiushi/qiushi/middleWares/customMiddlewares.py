from scrapy.contrib.downloadermiddleware.useragent import UserAgentMiddleware
#自定义中间件
class CustomUserAgent(UserAgentMiddleware):
    def process_request(self, request, spider):
        ua="Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1"
        request.headers.setdefault('User-Agent',ua)

class CustomProxy(object):
    def process_request(self,request,spider):
        request.meta['proxy']='http://202.85.213.220:3128'