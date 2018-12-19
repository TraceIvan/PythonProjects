#辗转相除法
print("示例：求12与32的最大公约数:")
a,b=32,12
while a%b:
    a,b=b,a%b
print("12和32的最大公约数为：%d\n"%b)
print("请输入两个整数：")
x=input()
y=input()
x1=int(x)
y1=int(y)
if x1<y1:
    x1,y1=y1,x1
while x1%y1:
    x1,y1=y1,x1%y1
print("最大公约数为：%d\n"%y1)
