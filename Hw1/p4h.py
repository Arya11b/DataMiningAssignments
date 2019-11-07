import p4data as d
from matplotlib import pyplot as plot
cu = list(set(d.class_name))
v_sw = [[] for c in cu]
v_sl = [[] for c in cu]
v_pw = [[] for c in cu]
v_pl = [[] for c in cu]
for i in range(len(d.class_name)):
    for j in range(len(cu)):
        if cu[j] == d.class_name[i]:
            v_sw[j].append(d.sepal_length[i])
            v_sl[j].append(d.sepal_width[i])
            v_pw[j].append(d.petal_length[i])
            v_pl[j].append(d.petal_width[i])
c = ['r','g','b']
m = ['o','^','x']
cl = [v_sl,v_sw,v_pl,v_pw]
cl_names = ['sepal length', 'sepal width','petal length', 'petal width' ]
index = 1
for i in range(len(cl)):
    for j in range(len(cl)):
        print(i,j)
        plot.subplot(4,4, index)
        index += 1
        for k in range(len(cu)):
            plot.scatter(cl[i][k], cl[j][k],c=c[k],marker=m[k])
            plot.xlabel(cl_names[i])
            plot.ylabel(cl_names[j])
plot.show()