#计算斐波那契数列小于参数n的所有值
def fib(n):
    a,b=1,1
    while a<n:
        print(a,end=' ')
        a,b=b,a+b
    print()

fib(1000)
#直接传值不会修改实参
def addone(a):
    print(a)
    a+=1
    print(a)
a=2
addone(a)
print(a)
#传递可变序列后修改操作会改变实参
def modify(v,item):
    v.append(item)
a=[2]
modify(a,3)
print(a)

#默认值参数
def say(message,times=1):
    print(((message+' ')*times)[0:-1])
print(say.__defaults__)#查看函数所有默认值参数的当前值，返回一个元组
say('hello')
say('hello',3)

def Join(List,sep=None):
    return((sep or ' ').join(List))
aList=['a','b','c']
print(Join(aList))
print(Join(aList,','))
#默认值参数必须出现在函数参数列表的最右端，任何一个默认值参数右边都不能出现非默认值参数
#默认值参数只被解释一次
def demo(newitem,old_list=None):#改为old_list=[]会出错
    if old_list is None:
        old_list=[]
    old_list.append(newitem)
    return old_list
print(demo('5',[1,2,3,4]))
print(demo('aaa',['a','b']))
print(demo('a'))
print(demo('b'))

#关键参数，使得实参顺序和形参顺序不一致
def demo2(a,b,c=5):
    print(a,b,c)
demo2(1,3)
demo2(3,1)
demo2(b=3,a=1)
demo2(c=8,a=9,b=0)

#可变长度参数
#*parametter接收任意多个实参并放入一个元组中
#**parametter接收类似关键参数一样显式赋值形式的多个实参并将其放入字典中
def demo3(*p):
    print(p)
demo3(1,2,3)
demo3(1,2,3,12,34,324)

def demo4(**p):
    for item in p.items():
        print(item)

demo4(x=1,y=2,z=3)

def demo5(a,b,c=4,*aa,**bb):
    print(a,b,c)
    print(aa)
    print(bb)
demo5(1,2,3,4,5,6,7,8,9,xx='1',yy='2',zz=3)
demo5(1,2,3,4,5,6,7,xx='1',yy='2',zz=3)

#参数传递序列解包
def fuc(a,b,c):
    print(a+b+c)
seq=[1,2,3]
fuc(*seq)#自动解包传递给多个单变量形参，使用字典默认键
tup=(1,2,3)
fuc(*tup)
dic={1:'a',2:'b',3:'c'}
fuc(*dic)
fuc(*dic.items())
fuc(*dic.values())
Set={1,2,23}
fuc(*Set)

#return 从一个函数返回
def maximum(x,y):
    return max(x,y)

#当一个变量已经在函数外定义，如果在函数内需要修改这个变量的值，并要将这个赋值结果反映到函数外，可以在函数内\
    # 使用global声明这个变量，明确声明使用同名的全局变量
#在函数内部直接将一个变量声明为全局变量，在函数外没有声明，在调用这个函数之后，将增加为新的全局变量
def fuc1():
    global x
    x=3
    y=4
    print(x,y)
x=5
fuc1()
print(x)
del x
fuc1()
print(x)