import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
class Squarify():
    def __init__(self):
        pass

    def normalize_sizes(self,sizes,dx,dy):
        total_size=sum(sizes)
        total_area=dx*dy
        sizes=list(map(float,sizes))
        sizes=list(map(lambda size:size*total_area/total_size,sizes))
        return sizes

    def pad_rectangle(self,rect):
        if rect['dx']>2:
            rect['x']+=1
            rect['dx']-=2
        if rect['dy']>2:
            rect['y']+=1
            rect['dy']-=2

    def layOutRow(self,sizes,x,y,dx,dy):
        covered_area=sum(sizes)
        width=covered_area/dy
        rects=[]
        for size in sizes:
            rects.append({'x':x,'y':y,'dx':width,'dy':size/width})
            y+=size/width
        return rects

    def layOutCol(self,sizes,x,y,dx,dy):
        covered_area=sum(sizes)
        height=covered_area/dx
        rects=[]
        for size in sizes:
            rects.append({'x':x,'y':y,'dx':size/height,'dy':height})
            x+=size/height
        return rects

    def layOut(self,sizes,x,y,dx,dy):
        return self.layOutRow(sizes,x,y,dx,dy) if dx>=dy else self.layOutCol(sizes,x,y,dx,dy)

    def leftOverRow(self,sizes,x,y,dx,dy):
        covered_area=sum(sizes)
        width=covered_area/dy
        leftOver_x=x+width
        leftOver_y=y
        leftOver_dx=dx-width
        leftOver_dy=dy
        return (leftOver_x,leftOver_y,leftOver_dx,leftOver_dy)

    def leftOverCol(self,sizes,x,y,dx,dy):
        covered_area=sum(sizes)
        height=covered_area/dx
        leftOver_x=x
        leftOver_y=y+height
        leftOver_dx=dx
        leftOver_dy=dy-height
        return (leftOver_x,leftOver_y,leftOver_dx,leftOver_dy)

    def leftOver(self,sizes,x,y,dx,dy):
        return self.leftOverRow(sizes,x,y,dx,dy) if dx>=dy else self.leftOverCol(sizes,x,y,dx,dy)

    def worst_ratio(self,sizes,x,y,dx,dy):
        return max([max(rect['dx']/rect['dy'],rect['dy']/rect['dx']) for rect in self.layOut(sizes,x,y,dx,dy)])

    def squarify(self,sizes,x,y,dx,dy):
        sizes=list(map(float,sizes))
        if len(sizes)==0:
            return []
        if len(sizes)==1:
            return self.layOut(sizes,x,y,dx,dy)
        #找到适合split的地方
        i=1
        while i<len(sizes) and self.worst_ratio(sizes[:i],x,y,dx,dy)>=self.worst_ratio(sizes[:(i+1)],x,y,dx,dy):
            i+=1
        current=sizes[:i]
        remaining=sizes[i:]
        (leftOver_x,leftOver_y,leftOver_dx,leftOver_dy)=self.leftOver(current,x,y,dx,dy)
        return self.layOut(current,x,y,dx,dy)+self.squarify(remaining,leftOver_x,leftOver_y,leftOver_dx,leftOver_dy)

    def padded_squarify(self,sizes,x,y,dx,dy):
        rects=self.squarify(sizes,x,y,dx,dy)
        for rect in rects:
            self.pad_rectangle(rect)
        return rects

if __name__=='__main__':
    SQRF=Squarify()
    x=0.0
    y=0.0
    width=950.0
    height=733.0
    norm_x=1000
    norm_y=1000

    fig=plt.figure()
    ax=fig.add_subplot(111)

    initvalues=[285.4,188.4,173,140.6,91.4,75.5,62.3,39.6,29.4,28.5,26.2,22.2]
    values=initvalues
    labels=["South Africa","Egypt","Nigeria","Algeria","Morocco","Angola","Libya","Tunisia","Kenya","Ethiopia",
            "Ghana","Cameron"]

    colors=[(214,27,31),(229,109,0),(109,178,2),(50,155,18),(41,127,214),(27,70,163),(72,17,121),(209,0,89),
            (148,0,26),(223,44,13),(195,215,0)]
    #将颜色转为[0,1]范围之间
    for i in range(len(colors)):
        r,g,b=colors[i]
        colors[i]=(r/255.0,g/255.0,b/255.0)
    #将values降序
    values.sort(reverse=True)
    #values的和必须与画的总面积相等
    values=SQRF.normalize_sizes(values,width,height)
    #填充的矩形在某些情况下可能会更直观
    rects=SQRF.padded_squarify(values,x,y,width,height)
    cmap=cm.get_cmap()

    color=[cmap(random.random()) for i in range(len(values))]
    x=[rect['x'] for rect in rects]
    y = [rect['y'] for rect in rects]
    dx = [rect['dx'] for rect in rects]
    dy = [rect['dy'] for rect in rects]

    ax.bar(x,dy,width=dx,bottom=y,align='edge',color=colors,label=labels)
    va='center'
    idx=1
    for l,r,v in zip(labels,rects,initvalues):
        x,y,dx,dy=r['x'],r['y'],r['dx'],r['dy']
        ax.text(x+dx/2,y+dy/2+10,str(idx)+"-->"+l,va=va,ha='center',color='white',fontsize=14)
        ax.text(x+dx/2,y+dy/2-12,"($"+str(v)+"b)",va=va,ha='center',color='white',fontsize=12)
        idx+=1
    ax.set_xlim(0,norm_x)
    ax.set_ylim(0,norm_y)
    plt.title('GDP rating of Africa')
    plt.show()