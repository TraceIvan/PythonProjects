ret=[1,0,1,1,1,1,1,0,1,0,1,0,0,1,0,0,0,0,0,1]
print("len:",len(ret))
tot_true=10
tot_neg=10
recall_k=[]#前k个检索结果中的相关文档在所有相关文档中的比率
precision_k=[]#前k个检索结果中的相关文档在前k个检索结果中的比率

cnt=0
AP=0.0
for i in range(len(ret)):
    cur=i+1
    if(ret[i]==1):
        cnt+=1
        recall_k.append(cnt/tot_true)
        precision_k.append(cnt/cur)
        AP+=cnt/cur
    else:
        recall_k.append(cnt/tot_true)
        precision_k.append(cnt/cur)
E_messure=[]
F_messure=[]
b=2
for i in range(len(ret)):
    F_messure.append(2/(1/recall_k[i]+1/precision_k[i]))
    E_messure.append((1+b*b)/(b*b/recall_k[i]+1/precision_k[i]))
print('R@K')
print(recall_k)
print('P@K')
print(precision_k)
print('F_messure\n',F_messure)
print('E_messure\n',E_messure)
print('AP:',AP/tot_true)



import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
xmajorLocator   = MultipleLocator(0.1) #将x主刻度标签设置为0.1的倍数
ymajorLocator   = MultipleLocator(0.1) #将y轴主刻度标签设置为0.1的倍数
import numpy as np
cur=(0.0,1.0)
fig=plt.figure()
fig.clf()
ax=plt.subplot(111)
print("recall_k:\n",recall_k)
precision_k=np.array(precision_k)
print("pre_k:\n",precision_k)
for i in range(len(recall_k)):
    cur_x=recall_k[i]
    if ret[i]==0:
        cur_y=cur[1]
    else:
        cur_y=max(precision_k[i:])
    plt.plot([cur[0],cur_x],[cur[1],cur_y],color='b',marker='o',markerfacecolor='r',linewidth=5)
    cur=(cur_x,cur_y)
plt.plot([cur[0],1.0],[cur[1],0.0],color='b',marker='o',markerfacecolor='r',linewidth=5)
plt.xlabel('Recall Rate')
plt.ylabel('Precision Rate')
plt.title('PR curve')
plt.xlim([0,1])
plt.ylim([0,1])
ax.xaxis.set_major_locator(xmajorLocator)
ax.yaxis.set_major_locator(ymajorLocator)
plt.show()




