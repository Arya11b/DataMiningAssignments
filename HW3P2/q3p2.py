from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('./drinks.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13:16].values
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7,test_size=0.3)
clf = MLPClassifier(hidden_layer_sizes=(8, 5), activation='tanh', solver='sgd', alpha=1e-5, random_state=1,
                    learning_rate='constant', learning_rate_init=0.7)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))
