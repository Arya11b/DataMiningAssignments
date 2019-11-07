f = open('iris.data.txt')
data = []
sepal_length = []
sepal_width = []
petal_length = []
petal_width = []
class_name = []
count = 0
for l in f:
    data.append(l)
data.pop()
for d in data:
    s = d[:-1].split(',')
    sepal_length.append(float(s[0]))
    sepal_width.append(float(s[1]))
    petal_length.append(float(s[2]))
    petal_width.append(float(s[3]))
    class_name.append(s[4])
count = len(class_name)