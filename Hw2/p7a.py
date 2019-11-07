import pandas as pd
import matplotlib
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def error_rate(pred,test):
    result = 0
    for i in range(len(pred)):
        result += 1 if pred[i] != test[i] else 0
    return result/len(pred)

names = ['Extra', 'Emoti', 'Agree', 'Consc', 'Openn', 'Win/Loss', 'Optimism', 'Pessimism', 'PastUsed', 'OwnPartyCount',
         'OppPartyCount', 'FutureUsed', 'PresentUsed', 'NumericContent']
dataset = pd.read_csv('./Q7_Data/US Presidential Data.csv', names=names)
x = dataset.iloc[1:, 1:].values
y = dataset.iloc[1:,0].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(error_rate(y_pred,y_test))
# part b
e = []
r = range(1,30)
for k in r:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    e.append(error_rate(y_pred,y_test))
plot.bar(r,e)
plot.xlabel('k')
plot.xticks(r)
plot.ylabel('error rate')
plot.show()