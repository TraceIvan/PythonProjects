import socket
HOST='169.254.122.68'#服务端主机IP地址
PORT=50007#服务端主机应用进程端口号
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect((HOST,PORT))#连接远程计算机
s.sendall(b'Hello,world')#发送数据
data=s.recv(1024)
print('Received ',repr(data.decode()))
s.close()