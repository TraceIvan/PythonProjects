class BinaryTree:
    def __init__(self,value=None):
        self.__left=None
        self.__right=None
        self.__data=value

    def insertLeftChild(self,value):
        if self.__left:
            print('left child tree already exists.')
        else:
            self.__left=BinaryTree(value)
            return self.__left

    def insertRightChild(self,value):
        if self.__right:
            print('right child tree already exists.')
        else:
            self.__right=BinaryTree(value)
            return self.__right

    def show(self):
        print(self.__data)

    def preOrder(self):
        print(self.__data)
        if self.__left:
            self.__left.preOrder()
        if self.__right:
            self.__right.preOrder()

    def postOrder(self):
        if self.__left:
            self.__left.preOrder()
        if self.__right:
            self.__right.preOrder()
        print(self.__data)

    def inOrder(self):
        if self.__left:
            self.__left.preOrder()
        print(self.__data)
        if self.__right:
            self.__right.preOrder()

if __name__=='__main__':
    print('please use me as a module')