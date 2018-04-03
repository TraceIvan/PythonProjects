import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.interpolate as spip

t=np.arange(0,2.5,0.1)
x=np.sin(2*np.pi*t)
y=np.cos(2*np.pi*t)

tcktuples,uarray=spip.splprep([x,y],s=0)#n维曲线的b样条插值

unew=np.arange(0,1.01,0.01)
splinevalues=spip.splev(unew,tcktuples)#求b样条或它的导数
plt.figure(1)
plt.plot(x,y,'x',splinevalues[0],splinevalues[1],np.sin(2*np.pi*unew),np.cos(2*np.pi*unew),x,y,'b')
plt.legend(['Linear','Cubic Spline','True'])
plt.axis([-1.25,1.25,-1.25,1.25])
plt.title('Parametric Spline Interpolation Curve')
plt.show()

