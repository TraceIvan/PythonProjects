print('列表比较')
"""比较"""
a=(1,2,3)#元组
b=(1,2,4)
print(a,':',type(a))
print(a,'<',b,a<b)
from operator import eq
print('a==b?',eq(a,b))

print('len、min、max、sum应用')
"""len、min、max、sum应用"""
a=[1,2,3,4]
print(len(a))
print(min(a))
print(max(a))
print(sum(a))
a=[1,2,[3,4],[5,6,[7]]]
print(len(a))
a={1:1 , 2:5 , 3:8}
print(max(a))
print(max(a.values()))
"""对字典，默认对键进行计算，可用values()说明对值进行计算"""
print(sum(a))
print(sum(a.values()))

print('zip应用')
"""zip可将多个列表或元组对应位置的元素组合为元组，返回zip对象"""
a=[1,2,3]
b=[4,5,6]
c=[7,8,9]
d=zip(a,b,c)
print(d)
e=list(d)
print(e)

print('enumerate应用')
"""enumerate应用"""
"""枚举列表、元组及其他可迭代对象，返回枚举对象，其中每个元素是包含下标和元素值的元组。对字典默认是对键操作"""
for item in enumerate(e):
    print(item)
for index,ch in enumerate('SDIBT'):
    print((index,ch),end=' ')
print()
a={1:1,2:5,3:8}
print(a,':')
for i,v in enumerate(a):
    print(i,v)
for i,v in enumerate(a.values()):
    print(i,v)