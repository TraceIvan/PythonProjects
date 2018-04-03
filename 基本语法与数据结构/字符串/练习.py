#把单独的字母i改为I
import re
example='i I i is Is his HIS hihi i'
pattern=re.compile(r'\bi\b')
print(pattern.sub('I',example))

#把单词中间的I改为i
import re
example='If implementatIon is explaIn it idea Imism. IsdIas is.'
pattern=re.compile(r'(\b\w*I\w*\b)')
index=0
while True:
    matchRe=pattern.search(example,index)
    if not matchRe:
        break
    str1=matchRe.group(0)
    str2=''
    index2=0
    pos=0
    while pos!=-1:
        pos=str1.find('I',index2)
        if pos==-1:
            break
        if pos==0 and (matchRe.start(0)<2 or example[matchRe.start(0)-2]=='.'):
            index2+=1
            continue
        str2=str2+str1[index2:pos]+'i'
        index2=pos+1
    str2=str2+str1[index2::]
    example=re.sub(str1,str2,example)
    index=matchRe.end(0)+1
print(example)
