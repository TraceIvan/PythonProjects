def searchPath(graph,start,end):
    result=[]
    __generatePath(graph,[start],end,result)
    result.sort(key=lambda x:len(x))
    return result
def __generatePath(graph,path,end,result):
    current=path[-1]
    if current==end:
        result.append(path)
    else:
        for n in graph[current]:
            if n not in path:
                __generatePath(graph,path+[n],end,result)

def showpath(result):
    print('the path from ',result[0][0],' to ',result[0][-1],' is :')
    for path in result:
        print(path)

if __name__=='__main__':
    graph={
        'A':['B','C','D'],
        'B':['E'],
        'C':['D','F'],
        'D':['B','E','G'],
        'E':['G'],
        'F':['D','G'],
        'G':['E','A','B']
    }
    r=searchPath(graph,'A','D')
    showpath(r)
