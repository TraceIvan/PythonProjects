import hashlib
import os
import sys

fileName=sys.argv[1]
if os.path.isfile(fileName):
    with open(fileName,'r') as fp:
        lines=fp.readlines()
    data=''.join(lines)
    print(hashlib.md5(data.encode()).hexdigest())
#ssdeep可以提供API函数计算文件模糊哈希值，比较俩文件的相似百分比
#暂不支持3.6