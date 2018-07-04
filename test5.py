import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# plot 
def build_integer(n, partition):
    list = []
    for i in range(0, partition):
        list.append(-float(n*(partition - i))/partition)
    list.append(0)
    for i in range(0, partition):
        list.append(float(n*(i+1))/partition)
    return list

def build_positive_integer(n, partition):
    list = []
    list.append(0)
    for i in range(0, partition):
        list.append(float(n*(i+1))/partition)
    return list

def build_integer(n, partition):
    list = []
    for i in range(0, partition):
        list.append(-float(n*(partition - i))/partition)
    list.append(0)
    for i in range(0, partition):
        list.append(float(n*(i+1))/partition)
    return list

n = 3
x = np.array(build_integer(n,100))
mu = 0
sigma = 1
yfuncequ = lambda t: (1/(np.sqrt(2*3.14159265358979)*sigma))*np.exp(-((t-mu)*(t-mu))/(2*sigma*sigma))
yfunc = np.vectorize(yfuncequ)
y = yfunc(x)

plt.plot(x, y, 'r-')

n = 3
x = np.array(build_integer(n,100))
mu = 0
sigma = 0.8
yfuncequ = lambda t: (1/(np.sqrt(2*3.14159265358979)*sigma))*np.exp(-((t-mu)*(t-mu))/(2*sigma*sigma))
yfunc = np.vectorize(yfuncequ)
y = yfunc(x)

plt.plot(x, y, 'b-')

n = 3
x = np.array(build_integer(n,100))
mu = 0
sigma = 0.6
yfuncequ = lambda t: (1/(np.sqrt(2*3.14159265358979)*sigma))*np.exp(-((t-mu)*(t-mu))/(2*sigma*sigma))
yfunc = np.vectorize(yfuncequ)
y = yfunc(x)

plt.plot(x, y, 'g-')

n = 3
x = np.array(build_integer(n,100))
mu = 0
sigma = 0.5
yfuncequ = lambda t: (1/(np.sqrt(2*3.14159265358979)*sigma))*np.exp(-((t-mu)*(t-mu))/(2*sigma*sigma))
yfunc = np.vectorize(yfuncequ)
y = yfunc(x)

plt.plot(x, y, 'y-')

n = 3
x = np.array(build_integer(n,100))
mu = 0
sigma = 0.4
yfuncequ = lambda t: (1/(np.sqrt(2*3.14159265358979)*sigma))*np.exp(-((t-mu)*(t-mu))/(2*sigma*sigma))
yfunc = np.vectorize(yfuncequ)
y = yfunc(x)

plt.plot(x, y, 'c-')

plt.show()

