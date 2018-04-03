print('列表推导式示例')
aList=[x*x for x in range(10)]
print(aList)
"""
上式相当于：
aList=[]
for x in range(10):
    aList.append(x*x)
print(aList)
"""
freshfruit=['banana','                  loganberry  ','  passion fruit   ']
aList=[w.strip() for w in freshfruit]
print(aList)
"""
上式等价于：
一：
"""
freshfruit=['banana','loganberry','passion fruit']
for i,v in enumerate(freshfruit):
    freshfruit[i]=v.strip()
print(aList)
"""
二：
"""
freshfruit=['banana','loganberry','passion fruit']
freshfruit=list(map(str.strip,freshfruit))
print(aList)

aList=[5 for i in range(10)]
print(aList)
aList=[i for i in range(10)]
print(aList)
bList=aList[::3]
print(bList)

"""使用列表推导式实现嵌套列表的平铺"""
vec=[[1,2,3],[4,5,6],[7,8,9]]
v2=[num for elem in vec for num in elem]
print(v2)
"""过滤不符合条件的元素"""
import os
fn=[filename for filename in os.listdir('.') if filename.endswith('.py')]#列出当前文件夹下所有的Python源文件
print(fn)

aList=[-1,-4,6,7.5,-2.3,9.-11]
bList=[i for i in aList if i>0]
print(bList)

scores={"张三":45,"李四":78,"王五":40,"周柳":96,"赵琪":65,"孙八":90,"张久":78,"吴氏":99,"东十一":60}
highest=max(scores.values())
lowest=min(scores.values())
print(highest)
print(lowest)
average=sum(scores.values())*1.0/len(scores)
print(average)
highestperson=[name for name,score in scores.items() if score==highest]
print(highestperson)

"""在推导式中使用多个循环，实现多序列元素的任意组合，并可结合条件语句过滤特定元素"""
v=[(x,y) for x in range(3) for y in range(3)]
print(v)
v2=[(x,y) for x in [1,2,3] for y in [3,1,4] if x!=y]
print(v2)

"""使用列表推导式实现矩阵转置"""
maxtri=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]
m2=[[row[i] for row in maxtri] for i in range(4)]
print(m2)
"""也可使用zip和list实现转置"""
m3=list(zip(*maxtri))
print(m3)