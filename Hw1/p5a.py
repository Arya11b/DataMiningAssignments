import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as la
from matplotlib import pyplot as plot
RAND = 10
# data stuff
a = np.load('data.npz')
# data imports
x1 = a['x1']
x2 = a['x2']
y = a['y']
# test data imports
x1t = a['x1_test']
x2t = a['x2_test']
yt = a['y_test']
b = [[1], [3], [2], [4]]
'''
# stochastic
for stochastic descent uncomment the following lines
'''
r_sample = random.sample(range(len(x1)),RAND)
x1_sample = []
x2_sample = []
y_sample = []
for r in r_sample:
    x1_sample.append(x1[r])
    x2_sample.append(x2[r])
    y_sample.append(y[r])
x1 = x1_sample
x2 = x2_sample
y = y_sample
'''
'''
ones = np.ones(len(x1))  # 1
firsts = np.multiply(x1, np.multiply(x2, x2))  # x1 * x2^2
seconds = np.multiply(x2, x2)
thirds = x1
X = np.column_stack((ones, firsts, seconds, thirds))
y = np.transpose([y])
alpha = 0.001
for i in range(400):
    derivation = np.matmul(np.matmul(np.transpose(X), X), b) - np.matmul(np.transpose(X), y)
    b = np.subtract(b, np.multiply(alpha, np.multiply(1 / la.norm(derivation, 2), derivation)))

onest = np.ones(len(x1t))  # 1
firstst = np.multiply(x1t, np.multiply(x2t, x2t))
secondst = np.multiply(x2t, x2t)
thirdst = x1t
Xt = np.column_stack((onest, firstst, secondst, thirdst))
yt = np.transpose([yt])
et = np.subtract(yt, np.matmul(Xt, b))
sset = la.norm(et,2)
print(sset)
# figure
fig = plot.figure()
ax = fig.add_subplot(111, projection='3d')

xs = x1t
ys = x2t
ax.scatter(xs, ys, yt, c='r', marker='o')
ax.scatter(xs, ys,np.matmul(Xt, b) , c='b', marker='^')
plot.show()
