
age=24
subject="计算机"
college="非重点"
if (age>25 and subject=="电子信息工程")or(college=="重点" and subject=="电子信息工程")or(age<=28 and subject=="计算机"):
    print("恭喜，你已获得我公司的面试机会！")
else:
    print("抱歉，你未达到面试要求")


import types
endFlag='yes'
s=0
while endFlag.lower()=='yes':
    x=input("请输入一个正整数：")
    if type(x)==int and 0<=int(x)<=100:
        s=s+int(x)
    else:
        print('不是数字或不符合要求')
    endFlag=input('继续输入？（yes or no）')

print('整数之和：',s)