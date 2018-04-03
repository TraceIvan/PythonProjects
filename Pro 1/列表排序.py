def print_list(alist):
    for i in alist:
        if type(i) is list:
            print_list(i)
        else:
            print(i,end=' ')
a=[3, 4, 5, 6, 7, 9, 11, 13, 15, 17]
import random

ta=[]
for i in range(20):
    ta.append(random.randint(1,10000))
print(ta)
tb=ta[:10]
tb.sort()
ta[:10]=tb
tb=ta[10:]
tb.sort(reverse=True)
ta[10:]=tb
print(ta)

random.shuffle(a)#打乱顺序
print_list(a)
print()

a.sort()#默认升序
print_list(a)
print()

a.sort(reverse=True)#降序排序
print_list(a)
print()
"""
lambda：这是Python支持一种有趣的语法，它允许你快速定义单行的最小函数，类似与C语言中的宏，这些叫做lambda的函数，是从LISP借用来的，可以用在任何需要函数的地方： 
>>> g = lambda x: x * 2 
>>> g(3) 
6 
>>> (lambda x: x * 2)(3) 
6


lambda表达式返回一个函数对象
例子：
func = lambda x,y:x+y
func相当于下面这个函数
def func(x,y):
    return x+y
 
注意def是语句而lambda是表达式
下面这种情况下就只能用lambda而不能用def
[(lambda x:x*x)(x) for x in range(1,11)]
"""
a.sort(key=lambda x:len(str(x)))#自定义排序
print_list(a)
print()
print(a)

"""sorted返回新列表，并不对原列表修改"""
a=[9, 7, 6, 5, 4, 3, 17, 15, 13, 11]
b=sorted(a)#默认升序
print(a, ' ', b)
print('id a:',id(a),'; id b:',id(b))
b=sorted(a, reverse=True)
print(a, ' ', b)
print('id a:',id(a),'; id b:',id(b))

"""逆序排列"""
a=[random.randint(50,100) for i in range(10)]
print(a)
a.reverse()
print(a)
"""reserved :不对原列表修改，返回逆序后的迭代对象"""
b=reversed(a)
print(b)
print(list(b))
print(b)
for i in b:#b为迭代对象，之前list(b)访问过后，已经迭代结束，不能再次迭代
    print(i,end=' ')
