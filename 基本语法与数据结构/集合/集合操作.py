a_set={8,9,10,11,12,13}
b_set={0,1,2,3,7,8}
print(a_set.union(b_set))#并集
print(a_set&b_set)#交集
print(a_set.intersection(b_set))#交集
print(a_set.difference(b_set))#差集：a-b
print(a_set.symmetric_difference(b_set))#对称差
print(a_set^b_set)#对称差
