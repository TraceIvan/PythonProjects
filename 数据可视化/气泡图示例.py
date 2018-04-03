import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
mov=pd.read_csv('ucdavis.csv')
x=mov.height
y=mov.momheight
z=mov.momheight

cm=plt.cm.get_cmap('RdYlBu')
fig,ax=plt.subplots(figsize=(12,10))
sc=ax.scatter(x,y,s=z*3,c=z,cmap=cm,linewidth=0.2,alpha=0.5)
ax.grid()
fig.colorbar(sc)
print(help(plt.scatter))
ax.set_xlabel('height',fontsize=14)
ax.set_ylabel('momheight',fontsize=14)
plt.show()