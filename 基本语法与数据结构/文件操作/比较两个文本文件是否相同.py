import difflib
import os
a=open('file2.txt','r')
b=open('test.txt','r')
contextA=a.read()
contextB=b.read()
s=difflib.SequenceMatcher(lambda x:x=="",contextA,contextB)
result=s.get_opcodes()
for tag,i1,i2,j1,j2 in result:
    print("%s contextA[%d:%d]=%s contextB[%d:%d]=%s"%\
          (tag,i1,i2,contextA[i1:i2],j1,j2,contextB[j1:j2]))
