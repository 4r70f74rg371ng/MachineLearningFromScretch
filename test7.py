import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def build_positive_integer(n, partition):
    list = []
    #list.append(0)
    for i in range(0, partition):
        list.append(float(n*(i+1))/partition)
    return list

n = 20
x = np.array(build_positive_integer(n,10000))
alpha = 3
numbda = 0.6
yfuncequ = lambda t: 1/t
yfunc = np.vectorize(yfuncequ)
y = yfunc(x)

plt.plot(x, y,'g-')
plt.show()
