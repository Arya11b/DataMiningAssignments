import numpy as np
import p4data as d
sl_mean = np.mean(d.sepal_length)
sw_mean = np.mean(d.sepal_width)
pl_mean = np.mean(d.petal_length)
pw_mean = np.mean(d.petal_width)
sl_var = np.var(d.sepal_length)
sw_var = np.var(d.sepal_width)
pl_var = np.var(d.petal_length)
pw_var = np.var(d.petal_width)
print('mean: ' , [sl_mean,sw_mean,pl_mean,pw_mean])
print('variance: ' , [sl_var,sw_var,pl_var,pw_var])
