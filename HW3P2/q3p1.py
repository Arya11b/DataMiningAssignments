import random
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - (tanh(x) * tanh(x))
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

class NeuralNetwork:
    def __init__(self, X,y,X_test,y_test):
        self.X = X
        self.y = y
        self.model = self.init_model([8, 5], 3)
        self.errors, no_iters = self.train( reg_lambda=0.03, learning_rate=0.1)
        z1, a1, z2, a2, z3, y_pred = self.feed_forward( X_test)
        for i in range(y_pred.shape[0]):
            for j in range(y_pred.shape[1]):
                if y_pred[i, j] >= 0.5:
                    y_pred[i, j] = 1
                else:
                    y_pred[i, j] = 0
        print(classification_report(y_test, y_pred))

    def init_model(self, hidden_layer_dimention, output_dimension):
        model = {}
        input_dim = self.X.shape[1]
        model['W1'] = np.random.randn(input_dim, hidden_layer_dimention[0]) / np.sqrt(input_dim)
        model['W2'] = np.random.randn(hidden_layer_dimention[0], hidden_layer_dimention[1]) / np.sqrt(hidden_layer_dimention[0])
        model['W3'] = np.random.randn(hidden_layer_dimention[1], output_dimension) / np.sqrt(hidden_layer_dimention[1])
        model['b1'] = np.zeros((1, hidden_layer_dimention[0]))
        model['b2'] = np.zeros((1, hidden_layer_dimention[1]))
        model['b3'] = np.zeros((1, output_dimension))
        return model

    def error_rate(self, X, y):
        z1, a1, z2, a2, z3, out = self.feed_forward(X)
        se = np.linalg.norm(out - y, axis=1, keepdims=True)
        sse = se.sum(axis=0, keepdims=True)
        return sse

    def train(self, reg_lambda=0.1, learning_rate=0.1):
        done = False
        i = 0
        self.errors = []
        while not done:
            z1, a1, z2, a2, z3, output = self.feed_forward(self.X)
            dW1, dW2, dW3, db1, db2, db3 = self.backprop( z1, a1, z2, a2, z3, output, reg_lambda)
            self.model['W1'] -= learning_rate * dW1
            self.model['b1'] -= learning_rate * db1
            self.model['W2'] -= learning_rate * dW2
            self.model['b2'] -= learning_rate * db2
            self.model['W3'] -= learning_rate * dW3
            self.model['b3'] -= learning_rate * db3
            if i % 100 == 0:
                error = self.error_rate(self.X, self.y)
                self.errors.append(error)
                print("Error after iteration %i: %f" % (i, error))
                if error < 0.15:
                    done = True
            i += 1
        return self.errors, i

    def backprop(self,z1, a1, z2, a2, z3, output, reg_lambda):
        delta3 = output - self.y
        dW3 = (a2.T).dot(delta3)
        db3 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(self.model['W3'].T) * (1 - np.power(a2, 2))
        dW2 = np.dot(a1.T, delta2)
        db2 = np.sum(delta2, axis=0)
        delta1 = delta2.dot(self.model['W2'].T) * (1 - np.power(a1, 2))
        dW1 = np.dot(self.X.T, delta1)
        db1 = np.sum(delta1, axis=0)
        dW3 += reg_lambda * self.model['W3']
        dW2 += reg_lambda * self.model['W2']
        dW1 += reg_lambda * self.model['W1']
        return dW1, dW2, dW3, db1, db2, db3

    def feed_forward(self,x):
        W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        a2 = np.tanh(z2)
        z3 = a2.dot(W3) + b3
        # soft max function
        exp_scores = np.exp(z3)
        out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return z1, a1, z2, a2, z3, out

dataset = pd.read_csv('./drinks.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13:16].values
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7,test_size=0.3)
n = NeuralNetwork(x_train,y_train,x_test,y_test)
