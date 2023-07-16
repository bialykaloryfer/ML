import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


s = 'C:/Users/piotr/Downloads/iris.data'
print(s)
df = pd.read_csv(s, header=None, encoding='utf-8')
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0, 2]].values

# plt.scatter(X[:50, 0], X[:50, 1], color="red", marker= 'o', label = 'Setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", marker= 'x', label = 'Vertisicolor')
# plt.xlabel('Długość działki [cm]')
# plt.ylabel('Długość płatka [cm]')
# plt.legend(loc= "upper left")
# plt.show()



class Perceptron(object):
    def __init__(self, eta = 0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        print(self.w_)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                print(xi, target)
                upgrade = self.eta * (target - self.predict(xi))
                self.w_[1:] += upgrade * xi
                self.w_[0] += upgrade
                errors += int(upgrade != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)



ppn = Perceptron(eta= 000.1, n_iter=15)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.show()