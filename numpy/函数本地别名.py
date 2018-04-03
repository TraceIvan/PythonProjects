import math
from time import time
import numpy as np
fastsin=math.sin

x=np.linspace(1,100,10000000)
y=np.linspace(1,100,10000000)
t1=time()
for i in range(10000000):
    x[i]=math.sin(x[i])
print('loop time of math.sin: %f'%(time()-t1))
t2=time()
for i in range(10000000):
    y[i]=fastsin(x[i])
print('loop time of fastsin: %f'%(time()-t2))
