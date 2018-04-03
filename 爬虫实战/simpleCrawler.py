import re
import urllib.request

class TodayMovie(object):
    '''获取金逸影城当日影视'''
    def __init__(self):
        self.url='http://www.33uu.com/'
        self.timeout=5
        self.fileName='./todayMoive.txt'
        '''内部变量定义完毕'''
        self.getMovieInfo()

    def getMovieInfo(self):
        response=urllib.request.urlopen(self.url,timeout=self.timeout)
        movieList=re.findall(b'<p class="name">.*?</p>',response.read())
        with open(self.fileName,'w',encoding='utf-8') as fp:
            for movie in movieList:
                movie=self.subStr(movie)
                print(movie.decode('utf-8'))
                fp.write(movie.decode('utf-8')+'\n')

    def subStr(self,st):
        st=st.replace(b'<p class="name">',b'')
        st=st.replace(b'</p>',b'')
        return st

if __name__=='__main__':
    tm=TodayMovie()