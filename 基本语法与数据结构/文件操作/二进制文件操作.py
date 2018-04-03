#pickle是较为常用并且速度非常快的二进制文件序列化模块
import pickle
f=open('sample.dat','wb')
n=7
i=13000000
a=99.056
s='中国人民 123abc'
lst=[[1,2,3],[4,5,6],[7,8,9]]
tu=(-5,10,8)
coll={4,5,6}
dic={'a':'apple','b':'banana','g':'grape','o':'orange'}
try:
    pickle.dump(n,f)
    pickle.dump(i,f)#把i转为字节串，并写入文件
    pickle.dump(a,f)
    pickle.dump(s,f)
    pickle.dump(lst,f)
    pickle.dump(tu,f)
    pickle.dump(coll,f)
    pickle.dump(dic,f)
except:
    print('写文件异常！')#如果写文件异常则跳到此处执行
finally:
    f.close()

f=open('sample.dat','rb')
n=pickle.load(f)
i=0
while i<n:
    x=pickle.load(f)
    print(x)
    i+=1
f.close()

#struct也是比较常用的对象序列化和二进制文件读写模块
import struct
n=1300000000
x=96.45
b=True
s=str('al#中国')
s=s.encode()#转为bytes格式
sn=struct.pack('if?',n,x,b)#把n、x、b对象转换为字节串
f=open('sample.dat','wb')
f.truncate()
f.write(sn)
f.write(s)#字符串bytes格式可直接写入
f.close()

f=open('sample.dat','rb')
sn=f.read(9)
tu=struct.unpack('if?',sn)
print(tu)
n=tu[0]
x=tu[1]
bl=tu[2]
print('n=',n)
print('x=',x)
print('bl=',bl)
s=f.read(9)
print(type(s))
s=bytes.decode(s)
f.close()
print('s=',str(s))
