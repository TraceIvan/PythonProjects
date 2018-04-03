#获取系统默认编码
import sys
print(sys.getdefaultencoding())
#显示指定字符串编码3种方式
#coding=utf-8
#coding:GBK
#-*-codeing:utf-8-*-
import string
s='中国'
print(s)
print(s.encode('GB2312'))
print(s.encode('GBK'))
print(s.encode('UTF-8'))
print(s.__class__)
