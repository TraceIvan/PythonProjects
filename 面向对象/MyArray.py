#filename:MyArray.py
#Function description: Array and its operating
#----------------------------------------------
import types

class MyArray:
    '''All the elements in this array must be numbers'''
    __value=[]
    __size=0

    def __IsNumber(self,n):
        if type(n)!=complex and type(n)!=float \
            and type(n)!=int:
            return False
        return True

    def __init__(self,*args):
        for arg in args:
            if not self.__IsNumber(arg):
                print('All elements must be numbers')
                return
        self.__value=[]
        for arg in args:
            self.__value.append(arg)
        self.__size=len(args)

    def __add__(self, n):
        if not self.__IsNumber(n):
            print('+ operating with ',type(n),' and number type is not supported.')
            return
        b=MyArray()
        for v in self.__value:
            b.__value.append(v+n)
        return b

    def __sub__(self, n):
        if not self.__IsNumber(n):
            print("- operating with ",type(n)," and number type is not supported.")
            return
        b=MyArray()
        for v in self.__value:
            b.__value.append(v-n)
        return b

    def __mul__(self,n):
       if not self.__IsNumber(n):
           print("* operating with ", type(n), " and number type is not supported.")
           return
       b = MyArray()
       for v in self.__value:
           b.__value.append(v * n)
       return b
#有问题  没有__div__
    def __truediv__(self, n):
        if not self.__IsNumber(n):
            print(r"/ operating with ", type(n), " and number type is not supported.")
            return
        b = MyArray()
        for v in self.__value:
            b.__value.append(v/n)
        return b

    def __floordiv__(self, n):
        if not self.__IsNumber(n):
            print(r"/ operating with ", type(n), " and number type is not supported.")
            return
        b = MyArray()
        for v in self.__value:
            b.__value.append(v//n)
        return b

    def __mod__(self,n):
        if not self.__IsNumber(n):
            print(r"% operating with ", type(n), " and number type is not supported.")
            return
        b = MyArray()
        for v in self.__value:
            b.__value.append(v%n)
        return b

    def __pow__(self,n):
        if not self.__IsNumber(n):
            print(r"** operating with ", type(n), " and number type is not supported.")
            return
        b = MyArray()
        for v in self.__value:
            b.__value.append(v**n)
        return b

    def __len__(self):
        return len(self.__value)

    #for:x
    #when use the object as a statement directly, the function will be called
    def __repr__(self):
        #equivalent to return 'self.__value'
        return repr(self.__value)#打印


    #for: print x
    def __str__(self):
        return str(self.__value)

    def append(self,v):
        if not self.__IsNumber(v):
            print("Only number can be appended.")
            return
        self.__value.append(v)
        self.__size+=1

    def __getitem__(self, index):
        if self.__IsNumber(index) and 0<=index<self.__size:
            return self.__value[index]
        else:
            print('Index is not a number or out of range.')

    def __setitem__(self, index, v):
        if self.__IsNumber(index) and 0<=index<self.__size:
            if self.__IsNumber(v):
                self.__value[index]=v
            else:
                print(v,' is not a number.')
        else:
            print(index,' is not a number or out of range.')

    #member test.support the keyword 'in'
    def __contains__(self, v):
        if v in self.__value:
            return True
        return False

    #dot product
    def dot(self,v):
        if not isinstance(v,MyArray):
            print(v, 'must be an instance of MyArray.')
            return
        if len(v)!=self.__size:
            print('The size must be equal.')
            return
        b=MyArray()
        for m,n in zip(v.__value,self.__value):
            b.__value.append(m*n)
        return sum(b.__value)

    #equal to
    def __eq__(self, v):
        if not isinstance(v,MyArray):
            print(v,' must be an instance of MyArray.')
            return
        from operator import eq
        if eq(self.__value,v.__value)==0:
            return True
        return False

    #less than
    def __lt__(self, v):
        if not isinstance(v,MyArray):
            print(v,' must be an instance of MyArray.')
            return
        from operator import eq
        if eq(self.__value, v.__value) < 0:
            return True
        return False

if __name__=='__main__':
    print('Please use me as a module.')