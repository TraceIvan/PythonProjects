import socket
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s.sendto(b"hello,world!",("169.254.122.68",5005))
#把string指定的内容发送给address指定的地址，包含一个接收方主机的IP地址和应用进程端口号的元组
s.close()