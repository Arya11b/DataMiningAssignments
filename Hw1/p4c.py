from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plot
import numpy as np
import p4data as d


def randrange(n, vmin, vmax):
    return (vmax - vmin)*np.random.rand(n) + vmin


cu = list(set(d.class_name))
v_sw = [[] for c in cu]
v_pw = [[] for c in cu]
for i in range(len(d.class_name)):
    for j in range(len(cu)):
        if cu[j] == d.class_name[i]:
            v_sw[j].append(d.sepal_length[i])
            v_pw[j].append(d.petal_width[i])
fig = plot.figure()
ax = fig.add_subplot(111, projection='3d')
n = d.count
c = ['r','g','b']
m = ['o','^','x']
for i in range(len(cu)):
    xs = v_sw[i]
    ys = v_pw[i]
    zs = randrange(len(v_sw[i]), 0, 100)
    ax.scatter(xs, ys, zs, c=c[i], marker=m[i])
ax.set_xlabel('sepal width')
ax.set_ylabel('petal width')

plot.show()