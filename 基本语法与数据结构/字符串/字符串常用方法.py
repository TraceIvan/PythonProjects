#1、find() 查找一个字符串在另一个字符串指定范围(默认是整个字符串)中首次出现的位置，不存在返回-1
s="apple,peach,banana,peach,pear"
print(s.find("peach"))
print(s.find("peach",7))
print(s.find("peach",7,20))


#2、split() 以指定字符为分隔符，将字符串分割成多个字符串，并返回包含分割结果列表
s="apple,peach,banana,peach,pear"
li=s.split(",")
print(li)
s="2014-10-31"
t=s.split("-")
print(t)
print(list(map(int,t)))
#如果不指定分隔符，则字符串中的任何空白字符(包括空格、换行符、制表符等)都会认为是分隔符
s="hello world \n\n my name\tis Dong         "
print(list(s.split()))
#该方法还允许指定最大分割次数
s="\n\nhello\t\t world\n\n\nMy name is Dong "
print(list(s.split()))
print(s.split(None,1))
print(s.split(None,2))
print(s.split(None,5))
print(s.split(None,6))

#3、join() 将列表中多个字符串进行连接，并在相邻两个字符串之间插入指定字符
li=["apple","peach","banana","pear"]
sep=","
print(sep.join(li))
print(''.join(li))
#join()运算符效率高于+
import timeit
strlist=['this is a long string that will not keep in memory.' for n in range(100)]
def use_join():
    return ''.join(strlist)
def use_plus():
    result=''
    for tstr in strlist:
        result=result+tstr
    return result
times=1000
jointimer=timeit.Timer('use_join()','from __main__ import use_join')
print('time for join:',jointimer.timeit(number=times))
plustimer=timeit.Timer('use_plus()','from __main__ import use_plus')
print('time for plus:',plustimer.timeit(number=times))


#4、lower(),upper(),capitalize(),title()，swapcase()
#将字符串转换为小写字符串；将字符串转换为大写字符串；将字符串首字母变为大写；将每个单词的首字母变为大写；大小写互换
s="What is Your Name?"
print(s.lower())
print(s.upper())
print(s.capitalize())
print(s.title())
print(s.swapcase())

#5、replace()：替换字符串中指定字符或子字符串的所有重复出现，每次只能替换一个字符或一个子字符串的重复出现
s="中国中国中国"
print(s)
s2=s.replace("中国","中华人民共和国")
print(s2)

#6、maketrans()，translate()
#maketrans()函数用来生成字符映射表；translate()则按照映射表关系转换字符串并替换其中的字符。可以同时处理多个不同的字符
#将字符串"abcdef123"转换为"uvwxyz@#$"
table=str.maketrans("abcdef123","uvwxyz@#$")
s="Python is a great programming language. I like it!"
print(s.translate(table))
#print(s.translate(table,"gtm"))#第二个参数表示转换后的字符串中要删除的字符,不可行

#7、strip()，rstrip()
#删除两端(或右端)的空白字符或连续的指定字符
s="    abs    "
s2=s.strip()
print(s2)
print("aaaaaaaaasssddfff".strip('a'))
print("aaaaaaaaasssddfff".strip('af'))
print("aaaaaaaaasssddfff".rstrip('a'))

#8、eval()
#尝试把任意字符串转化为Python表达式并求值
print(eval("3+4"))
a=3
b=4
print(eval('a+b'))
import math
print(eval('help(math.sqrt)'))
print(eval('math.sqrt(3)'))
s1="__import__('os').startfile(r'C:\windows\\notepad.exe')"
s2="__import__('os').system('md testtest')"
#eval(s1)
#eval(s2)

#9、in，not in
#判断一个字符串是否出现在另一个字符串当中
print('a' in 'abcdf')
print("ab" in 'abcdf')
print('j' in 'zxcb')

#10、startswith()，endswith()
#判断字符串是否以指定字符串开始或结束
print("abc".endswith('fd'))
print("abc".endswith('c'))
print("abc".startswith('fd'))
print("abc".startswith('a'))
print("abc".startswith(''))
print("abc".endswith(''))

#11、isalnum(),isalpha(),isdigit()
#测试字符串是否为数字或字母，是否为字母，是否为数字字符
print('12345asd'.isalnum())
print('12345asd'.isalpha())
print("12345asd".isdigit())
print("abcd".isalpha())
print('12324.0'.isdigit())
print("134".isdigit())




