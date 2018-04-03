#设计三维向量类,实现向量的加法、减法以及向量与标量的乘法和除法运算
class TD:
    __value=[]
    def __init__(self,v=None):
        if type(v)==None:
            self.__value=[0,0,0]
            return
        if type(v)!=list and type(v)!=tuple and type(v)!=iter:
            print('import must be a list or tuple or iter.')
            return
        if len(v)>3:
            print('length must not be greater than 3')
            return
        t=list(v)
        for i in t:
            if type(i)!=int and type(i)!=float and type(i)!=complex :
                print('the element must be a number.')
                return
        self.__value=t
        while len(self.__value)<3:
            self.__value.append(0)

    def __repr__(self):
        return repr(self.__value)\

    def __getitem__(self, index):
        if type(index)!=int or index<0 or index>2:
            print('index is not an integer or out of range')
            return
        return self.__value[index]

    def __setitem__(self,index, v):
        if type(index)!=int or index<0 or index>2:
            print('index is not an integer or out of range')
            return
        if type(v)!=int and type(v)!=float and type(v)!=complex:
            print('value must be a number.')
            return
        self.__value[index]=v

    def __str__(self):
        return str(self.__value)

    def __add__(self, v):
        if type(v)!=TD:
            print('the type must be TD.')
            return
        tmp=TD()
        for i,j in self.__value,v.__value:
            tmp.__value.append(i+j)
        return tmp

    def __sub__(self, v):
        if type(v)!=TD:
            print('the type must be TD.')
        tmp=TD()
        for i,j in self.__value,v.__value:
            tmp.__value.append(i-j)
        return tmp

    def __mul__(self, v):
        if type(v)!=int and type(v)!=float and type(v)!=complex:
            print('value must be a number.')
            return
        tmp=0
        for i in self.__value:
            tmp+=i*v
        return tmp
#标量除法

