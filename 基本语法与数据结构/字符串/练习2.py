#连续单词保留一个
import re
from operator import eq
pattern=re.compile(r'(\b\w+\b)')
example='This is is a a desk.'
ans=[]
index=0
while True:
    matchre=pattern.search(example,index)
    if not matchre:
        break
    tstr=matchre.group(0)
    tindex=matchre.end(0)+1
    tmatch = pattern.search(example, tindex)
    if not tmatch:
        ans.append(tstr)
        break
    tmp=tmatch.group(0)
    while eq(tmp,tstr):
        tindex=tmatch.end(0)+1
        tmatch=pattern.search(example,tindex)
        tmp=tmatch.group(0)
    ans.append(tstr)
    index=tindex
print(' '.join(ans))

#2
example='This is is a a desk.'
lstr=example.split(' ')
ans=[]
tot=len(lstr)
i=0
while i <tot:
    str1=lstr[i]
    j=i+1
    while j<tot and lstr[j]==str1:
        j+=1
    ans.append(str1)
    i=j
print(' '.join(ans))
