from urllib.request import urlopen
from urllib.error import HTTPError,URLError
from bs4 import BeautifulSoup
def getTitle(url):
    try:
        html=urlopen(url)
    except (HTTPError,URLError) as e:
        return None
    try:
        bsObj=BeautifulSoup(html.read(),'lxml')#指定解析器
        title=bsObj.body.h1
        if title==None:
            title=bsObj.title
    except AttributeError as e:
        return None
    return title

if __name__=='__main__':
    title=getTitle('http://www.baidu.com')
    if title==None:
        print("Title could not be found.")
    else:
        print(title)