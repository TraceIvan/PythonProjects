import heapq
import random
data=[i for i in range(10)]
print(data)
print(random.choice(data))#随机选择
print(random.choice(data))

random.shuffle(data)#乱序
print(data)

heap=[]
for n in data:
    heapq.heappush(heap,n)
print(heap)

heapq.heappush(heap,0.5)
print(heap)
u=heapq.heappushpop(heap,0.1)#先把item加入到堆中，然后再pop
print(u)

hh=heapq.merge([1,3,5,7],[0,2,4,8],[5,10,15,20],[],[25])#将多个列表合并，并进行堆调整，返回的是合并后的列表的迭代器
hh=[i for i in hh]
print(hh)

print(heapq.heappop(heap))#弹出最小的元素

myheap=[2,3,1,5,9,7,8,4,10,333]
heapq.heapify(myheap)#以线性时间将一个列表转化为堆，默认最小堆
u=heapq.heapreplace(myheap,6)#先pop，然后再把item加入到堆中
print(u,myheap)
# heapq.nlargest(n, iterable[, key])
# heapq.nsmallest(n, iterable[, key])
print(heapq.nlargest(3,myheap))#返回前3个最大的元素列表
print(heapq.nsmallest(3,myheap))#返回前3个最小的元素列表

portfolio = [
    {'name': 'IBM', 'shares': 100, 'price': 91.1},
    {'name': 'AAPL', 'shares': 50, 'price': 543.22},
    {'name': 'FB', 'shares': 200, 'price': 21.09},
    {'name': 'HPQ', 'shares': 35, 'price': 31.75},
    {'name': 'YHOO', 'shares': 45, 'price': 16.35},
    {'name': 'ACME', 'shares': 75, 'price': 115.65}
]
cheap = heapq.nsmallest(3, portfolio, key=lambda s: s['price'])
print(cheap)
expensive = heapq.nlargest(3, portfolio, key=lambda s: s['price'])
print(expensive)
#其他求top N问题方法
#N==1：
nums = [1,8,2,23,7,-4,18,23,42,37,2]
print(max(nums),min(nums))
#N较大：
print(sorted(nums)[:3])
print(sorted(nums,reverse=True)[:3])

