import socket
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
#family:socket.AF_INET表示IPV4，socket.AF_INET6 表示IPV6
#type:#'SOCK_STREAM'表示TCP，SOCK_DGRAM 表示UDP
s.bind(("",5005))#空字符串表示本机任何可用IP地址
data,addr=s.recvfrom(1024)#缓冲区大小为1024B，接收数据
print('received message :%s'%data.decode())#显示接收到的内容
s.close()
