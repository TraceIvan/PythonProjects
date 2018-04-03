import os
print(os.path.exists('test1.txt'))#判断文件是否存在
#os.rename('file1.txt','file\\file.txt')#实现重命名和移动
path=os.path.abspath('file\\file.txt')#返回绝对路径
print(os.path.dirname(path))#返回目录的路径
print(os.path.split(path))#对路径进行分割
print(os.path.splitdrive(path))#从路径中分割驱动器的名称
print(os.path.splitext(path))#从路径中分割文件的扩展名
print(os.path.getatime(path))#返回文件的最后访问时间
print(os.path.getctime(path))#返回文件的创建时间
print(os.path.getmtime(path))#返回文件的最后修改时间
print(os.path.getsize(path))#返回大小
print(os.path.isabs(path),os.path.isdir(path),os.path.isfile(path))#判断是否是绝对路径、目录、文件

#列出当前目录下所有扩展名为py的文件
print([fname for fname in os.listdir(os.getcwd()) if os.path.isfile(fname) and fname.endswith('.py')])
#listdir()返回path目录下的文件和目录列表
#getcwd()返回当前工作目录
#fstat()返回打开文件的所有属性
#stat()返回文件的所有属性
#remove()删除指定的文件
#access(path,mode)按照mode指定的权限访问文件

#将当前目录的所有扩展名为html的文件修改为htm
file_list=os.listdir(".")
for filename in file_list:
    try:
        pos=filename.rindex(".")
        if filename[pos+1:]=="html":
            newname=filename[:pos+1]+"htm"
            os.rename(filename,newname)
            print(filename+"更名为："+newname)
    except ValueError:
        continue

#另外还有shutil模块等


