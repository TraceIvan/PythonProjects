from meijunew.MiddleWares.resource import Proxys
import random

class RandomProxy(object):
    def process_request(self,request,spider):
        proxy=random.choice(Proxys)
        request.meta['proxy']='http://%s'%proxy
