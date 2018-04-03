'''
python中Trie实现：suffix_tree、pytire、trie、datrie等
'''

#寻找一个单词是否在Trie树中出现
_end='_end_'
def in_trie(trie,word):
    current_dict=trie
    for letter in word:
        if letter in current_dict:
            current_dict=current_dict[letter]
        else:
            return False
    else:
        if _end in current_dict:
            return True
        else:
            return False

#建立Trie树
def create_trie(*words):
    root=dict()
    for word in words:
        current_dict=root
        for letter in word:
            current_dict=current_dict.setdefault(letter,{})
        current_dict=current_dict.setdefault(_end,_end)
    return root

def insert_word(trie,word):
    if in_trie(trie,word):
        return
    current_dict=trie
    for letter in word:
        current_dict=current_dict.setdefault(letter,{})
    current_dict=current_dict.setdefault(_end,_end)

def remove_word(trie,word):
    current_dict=trie
    for letter in word:
        current_dict=current_dict.get(letter,None)
        if current_dict is None:#字典树中没有这个单词
            break
    else:
        del current_dict[_end]

dict=create_trie('foo','bar','baz','barz','bar')
print(dict)
print(in_trie(dict,'bar'))
print(in_trie(dict,'bars'))
insert_word(dict,'bars')
print(dict)
print(in_trie(dict,'bars'))