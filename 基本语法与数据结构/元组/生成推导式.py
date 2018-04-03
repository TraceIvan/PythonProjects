"""
1、生成器推导式的结果为一个生成器，而不是列表或元组
2、使用生成器对象的元素时，需要将其转化为列表或元组，或者使用next()方法(python 2.X版本)或__next__()(python 3.X版本)遍历，
或者直接将其作为迭代器使用。
3、无论哪种方法访问其元素，当所有元素访问结束后，如果要重新访问，必须重新创建该生成器对象
"""
g=((i+2)**2 for i in range(10))#生成器推导式用(),列表推导式用[]
print(g)
t1=tuple(g)
print(t1)
t2=tuple(g)
print(t2)

g=((i+2)**2 for i in range(10))
L1=list(g)
print(L1)
L2=list(g)
print(L2)

g=((i+2)**2 for i in range(10))
for i in range(10):
    print(g.__next__(),end=' ')
print()

g=((i+2)**2 for i in range(10))
for i in g:
    print(i,end=' ')
print()
