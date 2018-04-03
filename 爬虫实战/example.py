import  re
proxy='http://221.229.252.98:9797'
try:
    print(proxy)
    proxyMatch = re.compile('http[s]?://[\d]{1,3}\.[\d]{1,3}\.[\d]{1,3}\.[\d]{1,3}:[\d]{1,5}$')
    re.search(proxyMatch, proxy).group()
except  AttributeError:
    print('匹配失败')