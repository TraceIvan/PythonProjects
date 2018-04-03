f=open('file1.txt','a+')
s='helloworld'
f.write(s)
f.close()
#上述代码可优化：
s='helloworld'
with open('file1,txt','a+') as f:
    f.write(s)
#使用上下管理关键字with可以自动管理资源，不论何种原因跳出with块，总能保证文件被正确关闭，并且可以在代码块执行完毕后自
#动还原进入该代码块时的现场

f=open('file1.txt','w+')
f.truncate()
s='SDIBT中国山东烟台'
f.write(s)
f.close()
f=open('file1.txt','r')
print(f.read(5))#读取当前文件指针开始的若干个字符
print(f.read(7))
f.seek(0)#将文件指针移到文件开头
print(f.read(8))
f.close()

#读取并显示文本文件的所有行
f=open('file2.txt','r')
while True:
    line=f.readline()#从文本文件中读取一行内容并返回
    if line=='':
        break
    print(line,end='')#line里面有换行符
f.close()
f=open('file2.txt','r')
li=f.readlines()
for line in li:
    print(line,end='')

#移动文件指针
s='中国山东烟台STIBT'
fp=open(r'D:\Python\文件操作\file1.txt','w')
fp.truncate()
fp.write(s)
fp.close()
fp=open('file1.txt','r')
print(fp.read(3))
fp.seek(2)#转到距文件开头两个字节的位置
print(fp.read(1))
fp.seek(13)
print(fp.read(1))
fp.seek(15)
print(fp.read(1))
fp.seek(3)#出错，中文字2个字节
print(fp.read(1))
