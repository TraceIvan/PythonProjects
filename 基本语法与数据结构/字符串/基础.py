"""驻留机制：对于短字符串，将其赋给多
个不同对象时，内存中只有一个副本。
但不适用长字符串
"""
a='1234'
b='1234'
print(id(a)==id(b))
a='1234'*50
b='1234'*50
print(id(a)==id(b))

#判断一个变量是否为字符串
a='1234'
b=1234
print(isinstance(a,str))
print(isinstance(b,str))
#字符串格式化
x=1235
s1="%o"%x
print(s1,type(s1))
s2="%d"%x
print(s2,type(s2))
s3="%x"%x
print(s3,type(s3))
s4='%e'%x
print(s4,type(s4))
print(chr(ord('3')+1))#ord()函数接收一个字符，返回其ascii值；chr()接收0~255内ascii数值返回对应的字符
print("%s"%654)
print('%d%c'%(65,65))#使用元组对字符串进行格式化，按位置进行对应
print('%d,%c'%(65,65))#使用元组对字符串进行格式化，按位置进行对应
#print("%d"%"555"),试图将字符串转换为整数进行输出，抛出异常
print(int("55"))
print("%s"%[1,2,3])
print(str([1,2,3]))
print(str(234))
#format()格式化
print("the number {0:,} in hex is: {0:#x}, the number {1} in oct is {1:#o}.".format(5555,55))
print("the number {0:,} in hex is: {0:x}, the number {1} in oct is {1:o}.".format(5555,55))
print("the number {0:,} in hex is: {0:%}, the number {1} in oct is {1:%}.".format(5555,55))
print("the number {1:,} in hex is: {0:#x}, the number {0} in oct is {0:#o}.".format(5555,55))
print("my name is {name}, my age is {age}, and my QQ is {qq}.".format(name="Dong Fuguo",age=37,qq="43213576"))
solution=(5,8,13)
print("X:{0[0]},Y:{0[1]},Z:{0[2]}".format(solution))
weather=[("Monday","rain"),("Tuesday","sunny"),("Wednesday","sunny"),("Thursday","rain"),("Friday","cloudy")]
formatter="weather of {0[0]} is {0[1]}.".format
for item in map(formatter,weather):#map()第一个参数接收一个函数名，第二个参数接收一个可迭代对象,并将函数作用在每一个元素上
    print(item)
