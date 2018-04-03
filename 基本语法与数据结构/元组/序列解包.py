# 1、实现对多个变量同时赋值
x, y, z = 1, 2, 3
print(x, y, z)
print((x, y, z))
v_tuple = (False, 3.5, 'exp')
(x, y, z) = v_tuple
print(x, y, z)
x, y, z = v_tuple
print(x, y, z)
#其对列表同样适用
aList=[1,2,3]
b,c,d=aList
#对字典，默认是键，item()指定为键-值，value()指定为值
s={'a':1,'b':2,'c':3}
b,c,d=s.items()
print(b)
b,c,d=s
print(b)
b,c,d=s.values()
print(b,c,d)

#2、可以同时遍历多个序列
keys=['a','b','c','d']
values=[1,2,3,4]
for k,v in zip(keys,values):
    print(k,v)

#3、pro 1 中对enumerate的示例，也采用序列解包的操作

#4、对字典的遍历
s={'a':1,'b':2,'c':3}
for k,v in s.items():
    print(k,v)

#5、调用函数时，在实参前面加上一个*也可进行序列解包，实现将序列中的元素值依次传给相同数量的形参，详见 obj.函数