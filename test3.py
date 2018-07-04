import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def build_integer(n, partition):
    list = []
    for i in range(0, partition):
        list.append(-float(n*(partition - i))/partition)
    list.append(0)
    for i in range(0, partition):
        list.append(float(n*(i+1))/partition)
    return list
	
fig = plt.figure()
fig.subplots_adjust(left=0.2, wspace=0.6)

n = 3
x = np.array(build_integer(n,100))
mu = 0
sigma = 1
yfuncequ = lambda t: (1/(np.sqrt(2*3.14159265358979)*sigma))*np.exp(-((t-mu)*(t-mu))/(2*sigma*sigma))
yfunc = np.vectorize(yfuncequ)
y = yfunc(x)

ax1 = fig.add_subplot(221)
ax1.plot(x, y)
#ax1.show()

n = 3
x = np.array(build_integer(n,100))
mu = 0
sigma = 10
yfuncequ = lambda t: (1/(np.sqrt(2*3.14159265358979)*sigma))*np.exp(-((t-mu)*(t-mu))/(2*sigma*sigma))
yfunc = np.vectorize(yfuncequ)
y = yfunc(x)

ax2 = fig.add_subplot(222)
ax2.plot(x, y)
#ax1.show()

speed = [4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25]
dist = [2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85]

ax3 = fig.add_subplot(223)
ax3.scatter(speed, dist)

n = 10
x = np.array(build_integer(n,100))
mu = 0
sigma = 10
yfuncequ = lambda t: 1/(1+np.exp(-t))
yfunc = np.vectorize(yfuncequ)
y = yfunc(x)

ax4 = fig.add_subplot(224)
ax4.plot(x, y)

plt.show()



