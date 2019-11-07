from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plot
import numpy as np
import p4data as d

fig = plot.figure()
ax = fig.add_subplot(111, projection='3d')
x = d.petal_length
y = d.sepal_length
hist, xedges, yedges = np.histogram2d(x, y,bins=(6,6))
xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
xpos = xpos.flatten()/2
ypos = ypos.flatten()/2
zpos = d.count
# print(zpos)

dx = xedges[1] - xedges[0]
dy = yedges[1] - yedges[0]
dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
plot.xlabel('petal length')
plot.ylabel('sepal length')
plot.show()
