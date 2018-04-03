"""
1、用{}作为界定符
2、同一集合元素不允许重复
"""
a={3,5}
a.add(7)
print(a)

#用set将其他可迭代对象转换为集合，自动滤过重复元素
a_set=set(range(8,14))
print(a_set)

b_set= {0, 1, 2, 3, 0, 1, 2, 3, 7, 8}
print(b_set)

#删除操作
#1、del删除整个集合
#2、pop()弹出并删除其中一个元素
a={1,4,2,3}
v=a.pop()
print(v)
print(a.pop())
a.add(2)
print(a)
#3、remove()删除指定元素
a.remove(3)
print(a)
#4、clear()清空
a.clear()

