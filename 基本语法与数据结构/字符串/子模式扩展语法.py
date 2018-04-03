import re
exampleString='''There should be one-and preferably only one-obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than right now.'''
pattern=re.compile(r'(?<=\w\s)never(?=\s\w)')#查找不在句子开头和结尾的单词
matchResult=pattern.search(exampleString)
print(matchResult.span())
pattern=re.compile(r'(?<=\w\s)never')#查找位于句子结尾的单词
matchResult=pattern.search(exampleString)
print(matchResult.span())
pattern=re.compile(r'(?:is\s)better(\sthan)')#查找前面是is的better than组合
matchResult=pattern.search(exampleString)
print(matchResult.span())
print(matchResult.group(0))
print(matchResult.group(1))
pattern=re.compile(r'\b(?i)n\w+\b')#查找以n或N字母开头的所有单词
index=0
while True:
    matchResult=pattern.search(exampleString,index)
    if not matchResult:
        break
    print(matchResult.group(0),':',matchResult.span(0))
    index=matchResult.end(0)

pattern=re.compile(r'(?<!not\s)be\b')#查找前面没有单词not的单词be
index=0
while True:
    matchResult=pattern.search(exampleString,index)
    if not matchResult:
        break
    print(matchResult.group(0),':',matchResult.span(0))
    index=matchResult.end(0)

pattern=re.compile(r'(\b\w*(?P<f>\w+)(?P=f)\w*\b)')#查找具有连续相同字母的单词
index=0
while True:
    matchResult=pattern.search(exampleString,index)
    if not matchResult:
        break
    print(matchResult.group(0),':',matchResult.group(2))
    index=matchResult.end(0)+1




