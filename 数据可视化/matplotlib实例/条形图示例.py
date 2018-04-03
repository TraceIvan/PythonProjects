import numpy as np
import matplotlib.pyplot as plt
plt.figure(1)
N=7
winnersPlot=(142.6,125.3,62.0,81.0,145.6,319.4,178.1)
ind=np.arange(N)#x坐标
width=0.35#条形宽度
ax=plt.subplot(111)
winners=ax.bar(ind,winnersPlot,width,color="#ffad00")
nomineesPlot=(109.4,94.8,60.7,44.6,116.9,262.5,102.0)
nominees=ax.bar(ind+width,nomineesPlot,width,color='#9b3c38')
#添加标签、题目、标记
ax.set_xticks(ind+width/2)
ax.set_xticklabels(('Best Picture','Director','Best Actor','Best Actress','Editing','Visual Effects','Cinematography'))
ax.legend((winners[0],nominees[0]),('Academy Award Winners','Academy Award Nominees'))

def autolabel(rects):
    for rect in rects:
        height=rect.get_height()
        hcap="$"+str(height)+"M"
        ax.text(rect.get_x()+rect.get_width()/2.,height,hcap,ha='center',va='bottom',rotation='vertical')
autolabel(winners)
autolabel(nominees)
plt.show()