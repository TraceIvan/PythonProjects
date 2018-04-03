#1、循环累加
s=0
for i in range(1,101):
    s+=i
else:#当循环自然结束时执行else语句
    print(s)

s=i=0
while i<=100:
    s+=i
    i+=1
else:
    print(s)

#这段代码演示了模块的三种不同调用方法,并比较各自的运行时间
import time
import math
start=time.time()
for i in range(10000000):
    math.sin(i)
print('Time Used:',time.time()-start)
from math  import  sin as bsin
start=time.time()
for i in range(10000000):
    bsin(i)
print('Time Used:',time.time()-start)
loc_sin=math.sin
start=time.time()
for i in range(10000000):
    loc_sin(i)
print('Time Used:',time.time()-start)

for i in range(5):
    print(id(i),':',i)
    i+=1
    print(id(i), ':', i)
