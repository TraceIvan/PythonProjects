#计算小于100的最大素数
for n in range(100,1,-1):#从100到2，每次减1
    for i in range(2,n):#从2到n-1,判断是否素数
        if n%i==0:
            break
    else:#如果没有break，循环正常结束则执行else语句，说明为素数
        print(n)
        break
#输出10以内的奇数
for i in range(10):
    if i%2==0:
        continue
    print(i)