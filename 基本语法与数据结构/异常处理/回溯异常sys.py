import sys
try:
    1/0
except:
    t=sys.exc_info()#返回一个三元组：type表示异常的类型，value/message表示异常的信息或参数，traceback包含调用栈信息的对象
    print(t)

#比较标准的异常跟踪和sys.exc_info()之间的区别，sys.exc_info()可以直接定位最终引发异常的原因，但难以直接确定引发异常的代码位置
def A():
    1/0
def B():
    A()
def C():
    B()
#C()
try:
    C()
except:
    r=sys.exc_info()
    print(r)




