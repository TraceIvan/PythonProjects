#通过文件头信息来判断文件类型
def is_gif(fname):
    f=open(fname,'rb')
    first4=f.read(4)
    first4=first4.decode()
    print(first4)
    f.close()
    return first4==('GIF8')
print(is_gif('cjhy_ws.gif'))