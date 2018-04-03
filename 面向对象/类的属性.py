class Test:
    def __init__(self,value):
        self.__value=value
    @property
    def value(self):#只读，无法修改和删除
        return self.__value
T=Test(3)
print(T.value)
#T.value=5#出错
T.v=5#动态增加新成员
print(T.v)
del T.v#动态删除成员
#del T.value#试图动态删除对象属性，失败

class Test2:
    def __init__(self,value):
        self.__value=value
    def __get(self):
        return self.__value
    def __set(self,v):
        self.__value=v
    value=property(__get,__set)
t=Test2(3)
print(t.value)
t.value=5#允许修改属性
print(t.value)
print(t._Test2__value)

class Test3:
    def __init__(self,value):
        self.__value=value
    def __get(self):
        return self.__value
    def __set(self,v):
        self.__value=v
    def __del(self):
        del self.__value
    value=property(__get,__set,__del)
    def show(self):
        print(self.__value)
t=Test3(3)
t.show()
print(t.value)
t.value=5
t.show()
del t.value#同时删除该属性和对应数据成员
#print(t.value)#出错，属性已经被删除
t.value=1#为对象动态添加属性和对应的私有数据成员
t.show()
print(t.value)
t.show()
