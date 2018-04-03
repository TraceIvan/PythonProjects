class ShortInputException(Exception):
    '''自定义异常类'''
    def __init__(self,length,atleast):
        Exception.__init__(self)
        self.length=length
        self.atleast=atleast

try:
    s=input('请输入-->')
    if len(s)<3:
        raise ShortInputException(len(s),3)
except EOFError:
    print('你输入一个结束标记EOF')
except ShortInputException as x:
    print('ShortInputException:输入的长度是%d,长度至少应该是%d'%(x.length,x.atleast))
else:
    print('没有异常发生。')

#如果自己编写的模块需要抛出多个不同但相关的异常，可以先创建一个基类，然后创建多个派生类分别表示不同的异常
class Error(Exception):
    pass
class InputError(Error):
    def __init__(self,expression,message):
        self.expression=expression
        self.message=message
class TransitionError(Error):
    def __init__(self,previous,next,message):
        self.previous=previous
        self.next=next
        self.message=message

while True:
    try:
        x=int(input('please input a number:'))
        break
    except ValueError:
        print('That was no a valid number.Try again...')

#try……except……else 如果try中代码抛出异常
a_list=['China','America','England','France']
print('请输入字符串序号')
while True:
    n=int(input())
    try:
        print(a_list[n])
    except IndexError:
        print('列表元素下标越界或格式不正确，请重新输入字符串序号')
    else:
        break

#多except结构，一旦某一个捕获异常，其他子句不会再执行
try:
    x=input('请输入被除数：')
    y=input('请输入除数：')
    z=x/y
except ZeroDivisionError:
    print('除数不能为0')
except TypeError:
    print('被除数和除数应为数值类型')
except NameError:
    print('变量不存在')
else:
    print(x,'/',y,'=',z)

#当捕获的多个异常能够用同一段代码处理时，可以将窑捕获的异常卸载一个元组里
import sys
try:
    f=open('myfile.txt')
    s=f.readline()
    i=int(s.strip())
except (OSError,ValueError,RuntimeError,NameError):
    pass

#try……expect……finally……  finally子句的内容无论是否发生异常都会执行
'''
try:
    f=open('test.txt','r')
    line=f.readline()
    print(line)
finally:
    #f.close()
    pass
'''
#如果try子句的异常没有被捕捉和处理，或者except、else、finally子句的代码出现异常，则会在finally执行完后抛出
#上一例中，如果文件不存在，则f.close（）会抛出异常
def divide(x,y):
    try:
        result=x/y
    except ZeroDivisionError:
        print('division by zero!')
    else:
        print('result is',result)
    finally:
        print('executing finally clause')
divide(2,1)
divide(2,0)
#divide('2','1')#出错

#使用带有finally子句的异常处理结构，应尽量避免在finally子句中使用return语句，否则会出现意料外的错误
def demo_div(a,b):
    try:
        return a/b
    except:
        pass
    finally:
        return -1
print(demo_div(1,0))
print(demo_div(1,2))
print(demo_div(10,2))