a_list=[1,1,1,1,1,1,1,1,1,1,1,1]
for i in a_list[::-1]:
    print(i,' ',id(i),'\n')
    if i==1:
        a_list.remove(i)

print(len(a_list))
for i in range(len(a_list)):
    print(a_list[i],' ')
print('\n')
