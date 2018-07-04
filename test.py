import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

box = dict(facecolor='yellow', pad=5, alpha=0.2)

fig = plt.figure()
fig.subplots_adjust(left=0.2, wspace=0.6)

# get data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y = df.iloc[0:100, 4].values
y = [-1 if v == 'Iris-setosa' else 1 for v in y]
X=df.iloc[0:100, [0, 2]].values

# plot data
ax1 = fig.add_subplot(221)
ax1.scatter(X[:50,0], X[:50,1], color='red', marker='o', label='setosa')
ax1.scatter(X[50:100,0], X[50:100,1], color='blue', marker='x', label='versicolor')
ax1.set_xlabel('petal length')
ax1.set_ylabel('sepal length')
ax1.legend(loc='upper left')
#ax1.show()

eta = 0.1
parameters = np.zeros(1+X.shape[1])
errors = []
n_iter = 10
resolution = 0.01

def my_dot(a,b):
   total = 0.0
   if (len(a) == len(b)):
      for i in range(0, len(a)):
         total += a[i] * b[i]
   return total

def get_hypothsis(xi, parameters):
   hypothsis = my_dot(xi, parameters[1:])+parameters[0]
   if hypothsis >= 0.0:
      hypothsis = 1
   else:
      hypothsis = -1
   return hypothsis

for times in range(0, n_iter):
   error = 0
   for xi, target in zip(X, y):
      hypothsis = get_hypothsis(xi, parameters)
      update = eta * (target - hypothsis)
      parameters[1:] += update*xi
      parameters[0] += update
      error += int(update != 0.0)
   errors.append(error)

"""
>>> parameters
array([-0.4 , -0.68,  1.82])
>>>
"""

# plot errors
ax2 = fig.add_subplot(222)
ax2.plot(range(1, len(errors)+1), errors, marker='o')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Number of misclassification')

# plot classifier
markers = ('o', 'x', 's', '^', 'v')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:len(np.unique(y))])
x1_min, x1_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
x2_min, x2_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
#Z = np.where((np.dot(np.array([xx1.ravel(), xx2.ravel()]).T, parameters[1:])+parameters[0]) >= 0.0,  1, -1)
Z = np.array([get_hypothsis([v1,v2], parameters) for v1, v2 in zip(xx1.ravel(), xx2.ravel())])
Z = Z.reshape(xx1.shape)

ax3 = fig.add_subplot(223)
ax3.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
ax3.set_xlim(xx1.min(), xx1.max())
ax3.set_ylim(xx2.min(), xx2.max())

name_label = {-1: 'setosa', 1: 'versicolor'}
for idx, cl in enumerate(np.unique(y)):
   ax3.scatter(x = X[y == cl, 0], y = X[y == cl, 1],
      alpha = 0.8, c = cmap(idx),
      marker =  markers[idx], label=name_label[cl])

ax3.set_xlabel('sepal length [cm]')
ax3.set_ylabel('petal length [cm]')
ax3.legend(loc='upper left')
plt.show()
