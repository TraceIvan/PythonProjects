#判断是否闰年
year=input('请输入一个年份：')
year=int(year)
if year%100!=0 and year%4==0 or year%400==0:
    print('YES')
else:
    print('No')

#随机生成50个整数，然后删除所有奇数
import random
a=[random.randint(1,999) for i in range(50)]
print(a)
for i in a[::-1]:
    if i%2!=0:
        a.remove(i)
print(a)
