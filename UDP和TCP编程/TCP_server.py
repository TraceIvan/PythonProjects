import socket
HOST=''#本机所有可用IP地址
PORT=50007
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.bind((HOST,PORT))#绑定地址
s.listen(1)#开始监听
print('Listening at port:',PORT)
conn,addr=s.accept()
print('Connected by ',addr)
while True:
    data=conn.recv(1024)
    if not data:
        break
    conn.sendall(data)
conn.close()
