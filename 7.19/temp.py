import math
from scipy import stats
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import polyfit, poly1d
from scipy.stats import linregress

x = np.array([0.6,0.9,1,1.1,1.4,1.6,1.9,2.0,2.1,2.4])
y =  np.array([1,1,1,1,1,2,2,2,2,2])

slope, intercept, r_value, p_value, stderr = linregress(x, y)
slope, intercept
xx = np.arange(0,3,0.01)
plt.scatter(x,y,s=10,alpha=0.9)
plt.plot(xx, slope * xx + intercept, 'r-')
plt.title('k_real=1, '+'k_fit='+str(slope))

