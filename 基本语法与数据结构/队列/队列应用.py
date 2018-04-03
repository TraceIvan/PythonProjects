import queue
#建立队列，先进先出FIFO，以及插入和弹出
q=queue.Queue()
q.put(0)
q.put(1)
q.put(2)
print(q)
print(q.get())
print(q.get())
print(q)
#建立队列，后进先出LIFO
q=queue.LifoQueue()
q.put(1)
q.put(2)
q.put(3)
print(q)
print(q.get())
print(q.get())

#deque双端队列基本操作
qu=queue.deque(['Eric','John','Michael'])
qu.append('Terry')
qu.appendleft('Graham')
print(qu)
print(qu.pop())
print(qu.popleft())
print(qu)

#优先级队列构造
