import p4data as d
import numpy as np
print(d.class_name)
v_sl = []
v_sw = []
v_pl = []
v_pw = []
for i in range(len(d.class_name)):
    if d.class_name[i] == 'Iris-virginica':
        v_sl.append(d.sepal_length[i])
        v_sw.append(d.sepal_width[i])
        v_pl.append(d.petal_length[i])
        v_pw.append(d.petal_width[i])
ans = []
cl = [v_sl,v_sw,v_pl,v_pw]
for i in range(len(cl)):
    for j in range(i+1,len(cl)):
        corr = np.corrcoef(cl[i], cl[j])
        print(corr[0][1])
        ans.append(corr[0][1])
print(ans)
# sepal length and petal length
