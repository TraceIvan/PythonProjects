import numpy as np
def random_choose(data,labels,size):
    tot=np.shape(data)[0]
    ids=set([])
    while len(ids)<size:
        i=np.random.randint(0,tot)
        ids.add(i)
    random_data=[]
    random_label=[]
    for i in ids:
        random_data.append(data[i])
        random_label.append(labels[i])
    return random_data,random_label
