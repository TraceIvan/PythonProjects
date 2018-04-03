import os
print(os.getcwd())#返回当前目录
#os.mkdir(os.getcwd()+'\\temp')#创建目录
os.chdir(os.getcwd()+'\\temp')#把path设为当前工作目录
print(os.getcwd())
os.mkdir(os.getcwd()+'\\test')
print(os.listdir('.'))
os.rmdir('test')#删除目录
print(os.listdir('.'))

def visitDir(path):
    if not os.path.isdir(path):
        print('Error:"',path,'"is not a directory or does not exist.')
        return
    for lists in os.listdir(path):
        sub_path=os.path.join(path,lists)
        print(sub_path)
        if os.path.isdir(sub_path):
            visitDir(sub_path)
visitDir('D:\\Python')

def visitDir2(path):
    if not os.path.isdir(path):
        print('Error:"',path,'"is not a directory or does not exist.')
        return
    list_dirs=os.walk(path)
    for root,dirs,files in list_dirs:
        for d in dirs:
            print(os.path.join(root,d))#获取完整路径
        for f in files:
            print(os.path.join(root,f))#获取文件绝对路径
visitDir2('D:\\Python')
