#lambda表达式可以用来声明匿名函数。
#只可以包含一个表达式，不允许包含其他复杂的语句，但在表达式中可以调用其他函数
f=lambda x,y,z:x+y+z
print(f(1,2,3))
g=lambda x,y=2,z=3:x+y+z
print(g(1))
L=[(lambda x:x**2),(lambda x:x**3),(lambda x:x**4)]
print(L[0](2),L[1](2),L[2](2))
D={'f1':(lambda :2+3),'f2':(lambda :2*3),'f3':(lambda :2**3)}
print(D['f1'](),D['f2'](),D['f3']())
L=[1,2,3,4,5]
print(list(map((lambda x:x+10),L)))
print(L)
def demo(n):
    return n*n
print(demo(5))
a_List=[1,2,3,4,5]
print(list(map(lambda x:demo(x),a_List)))
data=list(range(20))
print(data)
import random
random.shuffle(data)
print(data)
data.sort(key=lambda x:x)
print(data)
data.sort(key=lambda x:len(str(x)))
print(data)
data.sort(key=lambda x:len(str(x)),reverse=True)
print(data)