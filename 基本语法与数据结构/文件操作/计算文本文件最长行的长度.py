f=open('file2.txt','r')
allLineLens=[len(line.strip()) for line in f]
f.close()
longest=max(allLineLens)
print(longest)

f=open('file2,txt','r')
longest=max(len(line.strip()) for line in f)
f.close()
print(longest)