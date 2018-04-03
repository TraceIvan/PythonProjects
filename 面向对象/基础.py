'''
面向对象三要素是：封装 继承 多态

'''

class Car:
    price=100000#定义类属型
    def __init__(self,c):
        self.color=c#定义实例属性
car1=Car("Red")
car2=Car("Blue")
print(car1.color,Car.price)
Car.price=110000#修改类属型
Car.name='QQ'#增加类属型
car1.color='Yellow'#修改实例属性
print(car2.color,Car.price,Car.name)
print(car1.color,Car.price,Car.name)
#如果类属性以两个下划线'_'开头则是私有属性，否则是公有属性，对于成员方法类似
#私有属性一般只能在类的公有成员方法中访问，即使python支持一种特殊的方式可以直接从外部访问类的私有成员，但不推荐
class A:
    def __init__(self,value1=0,value2=0):
        self._value1=value1
        self.__value2=value2
    def setValue(self,value1,value2):
        self._value1=value1
        self.__value2=value2
    def show(self):
        print(self._value1)
        print(self.__value2)
a=A()
print(a._value1)
print(a._A__value2)#在外部访问对象的私有数据成员
#_xxx：保护变量，不能用'from module import *'导入，只有类对象和子类对象能访问
#__xxx__:系统定义的特殊成员
#__xxx:类中私有成员，只有类对象自己访问，子类对象不能访问，但在对象外部可以使用'对象名._类名__xxx'访问
#IDLE下，一个'_'表示解释器中最后一次显示的内容或最后一次语句正确执行的输出结果

#公有方法通过对象名直接调用，私有方法不能通过对象名直接调用，只能在属于对象的方法中通过self调用或在外部通过pyhton支持的
#特殊方式调用；静态方法和类方法都可以通过类名和对象名调用，但不能访问属于对象的成员，只能访问属于类的成员。
class Root:
    __total=0#类私有成员
    def __init__(self,v):#构造函数
        Root.__total+=1
        self.__value=v#对象私有成员
    def show(self):
        print('self.__value:',self.__value)
        print('Root.__total:',Root.__total)
    @classmethod
    def classShowTotal(cls):#类方法
        print(cls.__total)
    @staticmethod
    def staticShowTotal():#静态方法
        print(Root.__total)
r=Root(3)
print(r.classShowTotal())#通过对象来调用类方法
print(r.staticShowTotal())#通过对象来调用静态方法
r.show()
rr=Root(3)
print(Root.classShowTotal())#通过类名调用类方法
print(Root.staticShowTotal())#通过类名调用静态方法
#Root.show()#试图通过类名直接调用实例方法，失败
Root.show(r)#可以通过这种方法来调用方法并访问实例成员
r.show()
