import numpy as np
import p4data as d
sl = d.sepal_length
sw = d.sepal_width
cov = np.cov(sl,sw)
print(cov)