"""
1、字典是键-值对的无需可变序列，每个元素的键和值用:分隔，相邻元素用，分隔，所有元素放在{}中
2、键：整数、实数、复数、字符串和元组等；但不能使用列表、集合、字典
3、键不允许重复，值可以重复
4、globals()返回和查看包含当前作用域内所有的全局变量和值的字典，locals()返回包含当前作用域内所有的局部变量和值的字典
"""
a=(1,2,3,4,5)
b='hello world.'
def demo():
    a=3
    b=[1,2,3]
    print('locals:',locals())
    print('globals:',globals())

demo()

#字典的创建与删除
#1、=赋值
a_dict={'server':'db.diveintopython3.org','database':'mysql'}
print(a_dict)
#2、dict()函数
keys=['a','b','c','d']
values=[1,2,3,4]
dic=dict(zip(keys,values))
print(dic)

#3、创建值为空的字典
adict=dict.fromkeys(['name','age','sex'])
print(adict)

#删除
#1、删除整个字典
del(adict)
a_dict.clear()
print(a_dict)
#2、删除指定键对应的元素
del(dic['b'])
print(dic)
val=dic.pop('c')#删除指定键的元素，并返回该对的值
print(dic,'\n',val)
tmp=dic.popitem()#随机删除某一对键值
print(dic,'\n',tmp)

#字典元素的读取
#1、以键作为下标访问,不存在时会抛出异常
adic={'name':'Dong','sex':'male','age':37}
print(adic)
print(adic['name'])
#2、更加安全的方式：get()获取指定键对应的值，并且可以指定键不存在时返回指定的值，如果不指定返回None
print(adic.get('adress'))
print(adic.get('adress','SDIBT'))
adic['score']=45
if adic.get('score')==None:
    adic['score']=[]
else:
    adic['score']=[adic['score']]
adic['score'].append(98)
adic['score'].append(97)
print(adic)

#3、items()返回键-值对列表，keys()返回键列表，values()返回值列表
aDic={'name':'Dong','sex':'male','age':37}
for item in aDic:
    print(item,end=' ')
print()
for key in aDic:
    print(key,end=' ')
print()
for key,value in aDic.items():
    print(key,value)
print(aDic.keys())
print(aDic.values())


#字典元素的操作
#1、修改键值:当键存在时修改其值，不存在时新建该键-值对
aDic['age']=38
print(aDic)
aDic['sdress']='SDIBP'
print(aDic)
#2、update()将另一个字典的键-值对一次性添加到当前字典对象
aDic.update({'a':'a','b':'b'})
print(aDic)