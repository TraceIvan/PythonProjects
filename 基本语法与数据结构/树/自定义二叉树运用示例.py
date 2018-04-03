import BinaryTree
root=BinaryTree.BinaryTree('root')
firstleft=root.insertLeftChild('A')
firstright=root.insertRightChild('B')
secondleft=firstleft.insertLeftChild('C')
thridright=secondleft.insertRightChild('D')
root.postOrder()
root.inOrder()