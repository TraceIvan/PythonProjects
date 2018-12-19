import numpy as np
import random
import re
from nltk.corpus import stopwords
import chardet
import base64 as beautifulsoup
import nltk
from math import log
s1="Jack loves playing basketball."
s2="Ann asks Jack for help."
s3="I help Ann fix the desk."
s4="Jack Ann"
s5="I fix"

words=set()
texts=[s1,s2,s3,s4]
vecs=[]
sz=len(texts)

for i in range(sz):
    tmp=texts[i]
    list_w=re.split('[\s\.!?,\"]+',tmp)#nltk.word_tokenize(tmp),re.split('\W+',tmp)
    if '' in list_w:
        list_w.remove('')
    words.update(list(list_w))
    vecs.append(list_w)

print(words)
print(vecs)
words=list(words)
words=sorted(words)
print(words)
sz2=len(words)
for i in range(sz):
    tmp=[]
    for j  in range(sz2):
        if words[j] in vecs[i]:
            tmp.append(1)
        else:
            tmp.append(0)
    print(tmp)
print('TF')
tf=[]
for i in range(sz):
    tmp=[]
    tot=len(vecs[i])
    for j  in range(sz2):
        cnt=vecs[i].count(words[j])
        tmp.append(cnt/tot)
    print(tmp)
    tf.append(tmp)
print('idf')
idf=[]
for i in range(sz2):
    cnt=0
    for j in range(sz):
        if words[i] in vecs[j]:
            cnt+=1
    idf.append(log(sz/cnt))
print(idf)
print('tf-idf')
tf_idf=[]
for i in range(len(tf)):
    tmp=[]
    for j in range(len(tf[i])):
        tmp.append(tf[i][j]*idf[j])
    print(tmp)
    tf_idf.append(tmp)
