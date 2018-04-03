import csv
import matplotlib.pyplot as plt

plt.figure(1)

with open('mortality1.csv') as csvf:
    mortdata=[row for row in csv.DictReader(csvf)]

x=[]
males_y=[]
females_y=[]
total_y=[]
first=True
for row in mortdata:
        x.append(int(row['Year']))
        males_y.append(float(row['Males']))
        females_y.append(float(row['Females']))
        total_y.append(float(row['Everyone']))

plt.plot(x,males_y,color='#1a61c3',label='Males',linewidth=1.8)
plt.plot(x,females_y,color='#bc108d',label='Females',linewidth=1.8)
plt.plot(x,total_y,color='#747e8a',label='Total',linewidth=1.8)
plt.legend(loc=0,prop={'size':10})
plt.show()