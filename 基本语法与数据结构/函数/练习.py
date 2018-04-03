#判断一个数是否为素数
def isPrime(x):
    from math import sqrt
    for i in range(2,int(sqrt(x)+1)):
        if x%i==0:
            return 'No'
    else:
        return 'Yes'
a=[34,12,324,345,65,6778,13,51,33,31]
for  i in a:
    print(isPrime(i))

#统计字符串中大写字母，小写字母，数字，其他字符的个数并用元组返回
def Clar(s):
    ans=[0,0,0,0]
    for i in s:
        if i>='A' and i<='Z':
            ans[0]+=1
        elif i>='a' and i<='z':
            ans[1]+=1
        elif i>='0' and i<='9':
            ans[2]+=1
        else:
            ans[3]+=1
    return tuple(ans)
print(Clar(r'''a1231']'s/n\''K.'1./\'2/\A/3\/HK;\'1;.2'N/3|'H\'1\i2?\3\'\KB4K5Jh45b6IUHBkmnb123kJ'';HI1236'''))

#验证局部变量是否会掩藏同名的全局变量——会
a=3
print(a)
def demo(x):
    a=x
    a+=2
    print(a)
demo(a)
print(a)

#编写函数，可以接收任意多个整数并输出其中的最大值和所有整数之和
def Cal(*ini):
    return(list((max(ini),sum(ini))))
print(Cal(123,23,4,5,46,89,56,7,2342,34,5,32,34))

#模拟内置函数sum
def Sum(x):
    sumv=0
    for i in x:
        sumv+=i
    return sumv
print(Sum([1,2,3,4]))
print(Sum((1,2,3,4)))

#模拟内置函数sorted()
def Sorted(x):
    ans=list(x)
    ll=len(ans)
    for i in range(ll):
        tv=ans[i]
        k=i
        for j in range(i+1,ll):
            if ans[j]<ans[k]:
                k=j
        if k!=i:
            ans[i],ans[k]=ans[k],ans[i]
    return ans
a=[2,3,1,56,32,56]
print(Sorted(a))
print(a)