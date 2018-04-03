from urllib.request import urlopen
from urllib.error import HTTPError,URLError
from bs4 import BeautifulSoup

def SimpleGetResponse(url):
    try:
        html=urlopen(url)
    except (HTTPError,URLError) as e:
        return None
    try:
        bsObj=BeautifulSoup(html.read(),'lxml')
        title=bsObj.body.h1
    except AttributeError as e:
        return None
    return bsObj