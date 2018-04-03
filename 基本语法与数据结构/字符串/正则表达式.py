import re
symbols=[',','+','-','/','//','**','>>','<<','+=','-=','*=','/=']
for i in symbols:
    patter=re.compile(r'\s*'+re.escape(i)+r'\s*')
    print(patter)

text='alpha.beta...gamma delta'
print(re.split('[\. ]+',text))
print(re.split('[\. ]+',text,maxsplit=2))
print(re.split('[\. ]+',text,maxsplit=1))
pat='[a-zA-Z]+'
print(re.findall(pat,text))

pat='{name}'
text='Dear {name}:welcome {name} here.'
print(re.sub(pat,'Mr.Dong',text))

s='a s d'
print(re.sub('a|s|d','good',s))

print(re.escape('http://www.python.org'))#字符串转义
print(re.match('done|quit','done'))
print(re.match('done|quit','done!'))
print(re.match('done|quit','doe!'))
print(re.match('done|quit','d!one!'))

#删除字符串中多余的空格
s='aaa     bbb    c d e   fff  '
print(re.sub('\s+',' ',s))
print(re.split('\s+',s))
print(re.split('[\s]+',s.strip()))
print(' '.join(re.split('[\s]+',s.strip())))
print(' '.join(re.split('\s+',s.strip())))
print(re.sub('\s+',' ',s.strip()))
print(s.split())
print(' '.join(s.split()))

example='ShanDong Institute of Business and Technology'
print(re.findall('\\ba.+?\\b',example))#以a开头的完整单词
print(re.findall('\\Bo.+?\\b',example))#不以o开头且含有o字母的单词剩余部分
print(re.findall('\\b\w.+?\\b',example))#所有单词
print(re.findall(r'\b\w.+?\b',example))#使用原始字符串，减少需要输入的符号数量
print(re.findall('\d+\.\d+\.\d+','python 2000.17.81'))#查找并返回x.x.x形式的数字
print(re.split('\s',example))#使用任何空白字符分割字符串

#match方法用于在字符串开头或指定位置进行搜索，模式必须出现在字符串开头或指定位置
#search方法用于在字符串整个范围中进行搜索
#findall方法用于在字符串中查找所有符合正则表达式的字符串并以列表返回
example='ShanDong Institute of Business and Technology'
pattern=re.compile(r'\bB\w+\b')#以B开头的单词
print(pattern.findall(example))
pattern=re.compile(r'\w+g\b')#以g结尾的单词
print(pattern.findall(example))
pattern=re.compile(r'\b[a-zA-Z]{3}\b')#查找3个字母长的单词
print(pattern.findall(example))
print(pattern.match(example))#从字符串开头开始匹配，不成功，没有返回值
print(pattern.search(example))#在整个字符串中搜哦，成功
pattern=re.compile(r'\b\w*a\w*\b')#查找所有含有字母a的单词
print(pattern.findall(example))

#sub()和subn()方法用于字符串替换
example='''Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.'''
pattern=re.compile(r'\bb\w*\b',re.I)#以字母b和B开头的单词（re.I:匹配时忽略大小写）
"""
编译标志
编译标志让你可以修改正则表达式的一些运行方式。在 re 模块中标志可以使用两个名字，一个是全名如 IGNORECASE，一个是缩写，一
字母形式如 I。（如果你熟悉 Perl 的模式修改，一字母形式使用同样的字母；例如 re.VERBOSE的缩写形式是 re.X。）多个标志可以通
过按位 OR-ing 它们来指定。如 re.I | re.M 被设置成 I 和 M 标志：
I 
IGNORECASE
使匹配对大小写不敏感；字符类和字符串匹配字母时忽略大小写。举个例子，[A-Z]也可以匹配小写字母，Spam 可以匹配 "Spam", 
"spam", 或 "spAM"。这个小写字母并不考虑当前位置。
L 
LOCALE
影响 "w, "W, "b, 和 "B，这取决于当前的本地化设置。
locales 是 C 语言库中的一项功能，是用来为需要考虑不同语言的编程提供帮助的。举个例子，如果你正在处理法文文本，你想用
 "w+ 来匹配文字，但 "w 只匹配字符类 [A-Za-z]；它并不能匹配 "é" 或 "?"。如果你的系统配置适当且本地化设置为法语，那么内
 部的 C 函数将告诉程序 "é" 也应该被认为是一个字母。当在编译正则表达式时使用 LOCALE 标志会得到用这些 C 函数来处理 "w 後
 的编译对象；这会更慢，但也会象你希望的那样可以用 "w+ 来匹配法文文本。
M 
MULTILINE
(此时 ^ 和 $ 不会被解释; 它们将在 4.1 节被介绍.)
使用 "^" 只匹配字符串的开始，而 $ 则只匹配字符串的结尾和直接在换行前（如果有的话）的字符串结尾。当本标志指定後， 
"^" 匹配字符串的开始和字符串中每行的开始。同样的， $ 元字符匹配字符串结尾和字符串中每行的结尾（直接在每个换行之前）。
S 
DOTALL
使 "." 特殊字符完全匹配任何字符，包括换行；没有这个标志， "." 匹配除了换行外的任何字符。
X 
VERBOSE
该标志通过给予你更灵活的格式以便你将正则表达式写得更易于理解。当该标志被指定时，在 RE 字符串中的空白符被忽略，除非该空
白符在字符类中或在反斜杠之後；这可以让你更清晰地组织和缩进 RE。它也可以允许你将注释写入 RE，这些注释会被引擎忽略；注释
用 "#"号 来标识，不过该符号不能在字符串或反斜杠之後。
"""
print(pattern.sub('*',example))#替换成'*'
print(pattern.sub('*',example,1))#只替换一次
pattern=re.compile(r'\bb\w*\b')
print(pattern.sub('*',example,1))

#split方法用于实现字符串分割
example=r'one,two,three,four/five\six?seven[eight]nine|ten'
pattern=re.compile(r'[,./\\?[\]\|]+')
print(pattern.split(example))
example=r'one1two2three3four4five5six6seven7eight8nine9ten'
pattern=re.compile(r'\d+')
print(pattern.split(example))
example=r'one two three four,five.six.seven,eight,nine9ten'
pattern=re.compile(r'[\s,.\d]+')
print(pattern.split(example))

#使用圆括号表示一个子模式，圆括号内的内容作为一个整体出现
telNumber='''Suppose my phone No. is 0535-1234567,
yours is 010-12345678,his is 025-87654321.'''
pattern=re.compile(r'(\d{3,4}-\d{7,8})')
print(pattern.findall(telNumber))

#match方法和search方法返回match对象，match对象的方法有group(),groups(),groupdict(),start(),end(),span()等
#group（）用来提出分组截获的字符串.group() 同group（0）就是匹配正则表达式整体结果,\
# group(1) 列出第一个括号匹配部分，group(2) 列出第二个括号匹配部分，group(3) 列出第三个括号匹配部分...
telNumber='''Suppose my phone No. is 0535-1234567,
yours is 010-12345678,his is 025-87654321.'''
pattern=re.compile(r'(\d{3,4})-(\d{7,8})')
index=0
while True:
    matchResult=pattern.search(telNumber,index)
    if not matchResult:
        break
    print('-'*30)
    print('Success:')
    for i in range(3):
        print('Searched content:',matchResult.group(i),\
              'Start from:',matchResult.start(i),'End at:',matchResult.end(i),\
              'Its span is:',matchResult.span(i)
              )
    index=matchResult.end(2)

