class Stack:
    def __init__(self,size=10):
        self._content=[]
        self._size=size

    def clear(self):
        self._content=[]

    def isEmpty(self):
        if not self._content:
            return True
        else:
            return False

    def push(self,v):
        if len(self._content)<self._size:
            self._content.insert(0,v)
        else:
            print('stack is full.')

    def pop(self):
        if len(self._content)>0:
            v=self._content[0]
            del(self._content[0])
            return v
        else:
            print('stack is empty.')

    def show(self):
        print(self._content)

s=Stack()
s.push(3)
s.show()
s.push(5)
s.show()
s.push(7)
s.show()
print(s.pop())
s.show()