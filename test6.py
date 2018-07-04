import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from cmath import sin, sqrt, pi, exp

p = [676.5203681218851
    ,-1259.1392167224028
    ,771.32342877765313
    ,-176.61502916214059
    ,12.507343278686905
    ,-0.13857109526572012
    ,9.9843695780195716e-6
    ,1.5056327351493116e-7
    ]

EPSILON = 1e-07  
def drop_imag(z):
    if np.abs(z.imag) <= EPSILON:
        z = z.real
    return z
    
def gamma(z):
    z = complex(z)
    if z.real < 0.5:
        y = pi / (np.sin(3.1415926*z) * gamma(1-z)) ### Reflection formula 
    else:
        z -= 1
        x = 0.99999999999980993
        for (i, pval) in enumerate(p):
            x += pval / (z+i+1)
        t = z + len(p) - 0.5
        y = np.sqrt(2*3.1415926) * t**(z+0.5) * np.exp(-t) * x
    return drop_imag(y)

def build_positive_integer(n, partition):
    list = []
    #list.append(0)
    for i in range(0, partition):
        list.append(float(n*(i+1))/partition)
    return list

n = 20
x = np.array(build_positive_integer(n,10000))
alpha = 1.5
numbda = 0.6
yfuncequ = lambda t: (np.power(numbda, alpha)*(np.power(t, alpha-1)*np.exp(-numbda*t)))/gamma(alpha)
yfunc = np.vectorize(yfuncequ)
y = yfunc(x)

plt.plot(x, y,'r-')

n = 20
x = np.array(build_positive_integer(n,10000))
alpha = 3
numbda = 0.6
yfuncequ = lambda t: (np.power(numbda, alpha)*(np.power(t, alpha-1)*np.exp(-numbda*t)))/gamma(alpha)
yfunc = np.vectorize(yfuncequ)
y = yfunc(x)

plt.plot(x, y,'g-')
plt.show()
