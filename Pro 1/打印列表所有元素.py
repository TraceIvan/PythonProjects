def print_list(alist):
    for i in alist:
        if type(i) is list:
            print_list(i)
        else:
            print(i)


def print_each_list(movies, level):
    for each_move in movies:
        if isinstance(each_move, list):
            print_each_list(each_move, level + 1)

        else:
            for num in range(level):
                print('\t', end='')
            print(each_move)


t = ['1', ['2', '3'], ['4', ['5', '6']]]
print_list(t)
print('\n')
print_each_list(t, 0)
print(t)