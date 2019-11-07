import pandas as pd
import matplotlib
import matplotlib.pyplot as plot
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred


def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))
    print("Accuracy : ",
          accuracy_score(y_test, y_pred) * 100)
    print("Report : ",
          classification_report(y_test, y_pred))

dataset = pd.read_csv('./Q8_Data/noisy_train.csv')
test_dataset = pd.read_csv('./Q8_Data/noisy_test.csv')
groups = dataset.iloc[:, 1:].values
classes = dataset.iloc[:,0].values
test_groups = test_dataset.iloc[:, 1:].values
test_classes = test_dataset.iloc[:, 0].values
print(test_groups)
print(test_classes)
# print(gini(classes))
enc = preprocessing.LabelEncoder()
h = len(groups)
w = len(groups[0])
th = len(test_groups)
tw = len(test_groups[0])
gt = groups.reshape(len(groups) * len(groups[0]),1)
enc.fit(gt)
gtt = enc.transform(gt)
ktt = gtt.reshape(h,w)
tgt = test_groups.reshape(len(test_groups) * len(test_groups[0]),1)
enc.fit(tgt)
tgtt = enc.transform(tgt)
tktt = tgtt.reshape(th,tw)
clf_gini = DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=9, min_samples_leaf=5)
clf_gini.fit(ktt,classes)
pred = prediction(tktt,clf_gini)
cal_accuracy(test_classes,pred)
