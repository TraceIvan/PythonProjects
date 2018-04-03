import csv
import matplotlib.pyplot as plt

plt.figure(1)
x=[]
y=[]
for i in range(0,7):
    y.append([])
with open('mortality2.csv') as csvf:
    csfr=csv.reader(csvf)
    for row in csfr:
        x.append(int(row[0]))
        for i in range(1, 8):
            y[i - 1].append(float(row[i]))
colorsdata=['#168cf8','#ff0000','#009f00','#1d437c','#eb912b','#8663ec','#38762b']
labeldata=['Below 25','25-44','45-54','55-64','65-74','75-84','Over 85']
for i in range(0,7):
    plt.plot(x,y[i],color=colorsdata[i],label=labeldata[i],linewidth=2)
plt.legend(loc=0,prop={'size':10})
plt.show()