def print_list(alist):
    for i in alist:
        if type(i) is list:
            print_list(i)
        else:
            print(i,end=' ')


#切取倒序
aList=[3,4,5,6,7,9,11,13,15,17]
bList=aList[::-1]
print_list(aList)
print()
print_list(bList)
print()
aList=[3,5,7]
print_list(aList)
print()
bList=aList[len(aList):]
print_list(bList)
print()
aList[len(aList):]=[9]
print_list(aList)
print()
aList[:3]=[1,2,3]
print_list(aList)
print()
aList[:3]=[]
print_list(aList)
print()
aList=list(range(10))
print_list(aList)
print()
aList[::2]=[0]*(len(aList)//2)
print_list(aList)
print()

aList=[3,5,7,9,11]
print_list(aList)
del aList[:3]
print_list(aList)
print()
"""切片返回浅复制"""
from operator import eq
aList=[3,5,7]
bList=aList#两者指向同一个内存
print_list(aList)
print()
print_list(bList)
print()
print('id a: ',id(aList),'; id b: ',id(bList))
print("a==b?",aList==bList,'; a is b? ',aList is bList,'; cmp(a,b):',eq(aList,bList))
bList[1]=8
print_list(aList)
print()
print_list(bList)
print()
print('id a: ',id(aList),'; id b: ',id(bList))
print("a==b?",aList==bList,'; a is b? ',aList is bList,'; cmp(a,b):',eq(aList,bList))

aList=[3,6,7]
bList=aList[::]
print_list(aList)
print()
print_list(bList)
print()
print('id a: ',id(aList),'; id b: ',id(bList))
print("a==b?",aList==bList,'; a is b? ',aList is bList,'; cmp(a,b):',eq(aList,bList))
bList[1]=8
print_list(aList)
print()
print_list(bList)
print()
print('id a: ',id(aList),'; id b: ',id(bList))
print("a==b?",aList==bList,'; a is b? ',aList is bList,'; cmp(a,b):',eq(aList,bList))