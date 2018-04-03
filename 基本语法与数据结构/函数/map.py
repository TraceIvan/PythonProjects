#map()可以将一个单参数函数依次作用到一个序列或迭代器对象的每一个元素上，并返回一个列表作为结果，该列表中的每一个元素
#是原序列中的元素经过函数处理后的结果，不对原序列或迭代器对象做任何修改
print(list(map(str,range(5))))
def add5(v):
    return v+5
print(list(map(add5,range(10))))
#reduce可以将一个接受2个参数的函数以累积的方式从左到右依次作用到一个序列或迭代器对象的所有元素上
seq=[1,2,3,4,5,6,7,8,9]
from functools import reduce
print(reduce(lambda x,y:x+y,seq))
def add(x,y):
    return x+y
print(reduce(add,range(10)))
print(reduce(add,map(str,range(10))))

#filter()将一个单参数函数作用在一个序列上，返回该序列中使得该函数返回值为True的那些元素组成的列表、元组或字符串
seq=['foo','x41','?!','***']
def func(x):
    return x.isalnum()
print(list(filter(func,seq)))
print(seq)
print([x for x in seq if x.isalnum()])
print(list(filter(lambda x:x.isalnum(),seq)))

#yield()创建生成器
def f():
    a,b=1,1
    while True:
        yield a
        a,b=b,a+b
a=f()
for i in range(100):
    print(a.__next__(),end=' ')
print()
for i in f():#惰性求值
    if i>100:
        break
    print(i,end=' ')
print()

#使用dis模块查看函数的字节码指令
def add1(n):
    n+=1
    return n
import dis
print(dis.dis(add))
