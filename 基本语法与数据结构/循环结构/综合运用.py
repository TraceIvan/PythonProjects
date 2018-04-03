#输出1+2+……+100的结果
sm=0
for i in range(101):
    sm+=i
print(sm)

print(sum(range(1,101)))

#输出序列中元素
a=['a','b','mpilgrim','z','example']
for i,v in enumerate(a):#enumerate会将数组或列表组成一个索引序列
    print('列表的第',i+1,'个元素是：',v)

#求1~100之间能被7整除，但不能同时被5整除的所有整数
for i in range(1,101):
    if i%7==0 and i%5!=0:
        print(i)
#输出水仙花数
for i in range(100,1000):
    if i==(i%10)**3+(i//10%10)**3+(i//100)**3:
        print(i)
#求平均分
score=[70,90,78,85,97,94,65,80]
s=0
for i in score:
    s+=i
print(s/len(score))
print(sum(score)/len(score))

#打印九九乘法表
for i in range(1,10):
    for j in range(1,i+1):
        print(i,'*',j,'=',i*j,end='\t')
    print()
#求200以内能被17整除的最大正整数
for i in range(200,1,-1):
    if i%17==0:
        print(i)
        break
#判断一个数是否素数
from math import sqrt
n=input('输入一个正整数：')
n=int(n)
m=int(sqrt(n)+1)
for i in range (2,m+1):
    if n%i==0:
        print('No')
        break
else:
    print('YES')