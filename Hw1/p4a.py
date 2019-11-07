import matplotlib
import matplotlib.pyplot as plot
import p4data as d
    # sepal_length
plot.hist2d(d.petal_length,d.sepal_length)
plot.xlabel('petal length')
plot.ylabel('sepal length')
plot.show()