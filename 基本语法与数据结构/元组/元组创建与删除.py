"""
定义元组时所有元素放在一对圆括号里
元组一旦创建，就不可对其进行修改
"""
a_tuple=('a')#如果只创建包含一个元素的元组，只把元素放在圆括号里是不行的，还需要加一个逗号
print(a_tuple)
a_tuple=('a',)
print(a_tuple)
print(len(a_tuple))
a_tuple=('a','b','mpilgrim','z','example')
print(a_tuple)

a=3
print(a,type(a))
a=3,
print(a,type(a))
a=1,2
print(a,type(a))

#类似list(),可用tuple()将其他类型序列转换为元组
aList=[-1,-4,6,7.5,-2.3,9,-11]
bstring='abcdef'
print(tuple(aList))
print(tuple(bstring))

#对于元组，因其不可变，所以只能用del命令删除整个元组对象
del(a)

#元组也能进行切片操作，但是只能只读，而不能修改
b_tuple=a_tuple[::-1]
print(b_tuple)
b_tuple=a_tuple[2:]
print(b_tuple)

#list()如果传入一个元组，则会返回一个列表，并且元素可变
#list()可以融化元组使其可变，tuple()可看作冻结列表并使其不可变

#元组的访问和处理速度比列表更快，如果只需要遍历而不做修改，建议使用元组

#元组作为不可变序列，和整数、字符串一样可以当作字典的键，而列表却不能，因其可变