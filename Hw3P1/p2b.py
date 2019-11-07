import sklearn
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import LinearSVC
def plot_data(X, y):
    class0 = np.where(y == -1)[0]
    plot.scatter(X[class0, 0], X[class0, 1], c='red', marker='s')
    class1 = np.where(y == 1)[0]
    plot.scatter(X[class1, 0], X[class1, 1], c='blue', marker='o')

def plot_margins(svm, X, y):
    xmin, xmax = plot.xlim()
    ymin, ymax = plot.ylim()
    # create grid to evaluate model
    xx = np.linspace(xmin, xmax, 30)
    yy = np.linspace(ymin, ymax, 30)
    XX, YY = np.meshgrid(xx, yy)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = svm.decision_function(xy).reshape(XX.shape)
    # plot decision boundary and margins
    plot.contour(XX, YY, Z, colors='black', levels=[-1, 0, 1], alpha=0.6, linestyles=['--', '-', '--'])

dataset = pd.read_csv('./svmdata2.csv')
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2].values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.4, random_state=1)
x1_train_c = []
x2_train_c = []
print(y)
for i in range(len(x_train)):
    if y[i] == -1:
        x1_train_c.append(x_train[i])
    else:
        x2_train_c.append(x_train[i])
## a
plot.scatter(x=x_train[:, 0], y=x_train[:, 1], c=['red' if l == -1 else 'blue' for l in y_train])
plot.show()

## c
c_val = 1.0
clf = svm.LinearSVC(C=c_val)
sm_svm = clf.fit(x_train, y_train)
y_pred = clf.predict(x_val)
print(clf)

count = 0
Cl = []
errors = []
while (c_val < 500):
    count = 0
    clf = LinearSVC(C=c_val)
    sm_svm = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_val)
    for i in range(len(y_pred)):
        if (y_pred[i] != y_val[i]):
            count += 1
    c_val += 1
    Cl.append(c_val)
    errors.append(float(count) / float(len(y_val)))

c_val = Cl[errors.index(min(errors))]
clf = LinearSVC(C=c_val)
sm_svm = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

x_min, x_max = min(x_test[:,0]) - 1, max(x_test[:,0]) + 1
y_min, y_max = min(x_test[:,1]) - 1, max(x_test[:,1]) + 1
h = 0.025
xd, yd = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = sm_svm.predict(np.c_[xd.ravel(), yd.ravel()])

Z = Z.reshape(xd.shape)
plot.figure(2)
plot.contourf(xd, yd, Z, cmap=plot.cm.coolwarm, alpha=0.6)

plot.scatter(x_test[:,0], x_test[:,1], cmap=plot.cm.coolwarm)
plot.xlim(xd.min(), xd.max())
plot.ylim(yd.min(), yd.max())
plot.xticks(())
plot.yticks(())
plot.show()

cnt = 0
for i in range(len(y_pred)):
    if (y_pred[i] != list(y_test)[i]):
        cnt += 1
clf_error = float(cnt) / float(len(y_test))
print("error rate " + clf_error.__str__())
# d
seed = 10
svm_nonlinear = svm.SVC(C=100, kernel='poly', degree=2, random_state=seed)
svm_nonlinear.fit(x_test, y_test)
plot_data(x_test, y_test)
plot_margins(svm_nonlinear, x_train, y_train)
plot.show()
