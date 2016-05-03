from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils import np_utils


def normal(n, m, sigma, mu):
    return sigma*np.random.randn(n, m) + mu

def label(X, l):
    return np.concatenate([X, l*np.ones((X.shape[0], 1))], 1)

def shuffled(X):
    np.random.shuffle(X)
    X_data = X[:, :-1]
    X_labels = X[:, -1:]
    return X_data, X_labels


l1 = 0
l2 = 1
lambda1 = -1
lambda2 = 1
sigma1 = 1
sigma2 = 1

# training data
n_tr = 1000
X_tr1 = label(normal(n_tr, 1, sigma1, lambda1), l1)
X_tr2 = label(normal(n_tr, 1, sigma2, lambda2), l2)
X_tr = np.vstack((X_tr1, X_tr2))
X_tr, y_tr = shuffled(X_tr)

dimof_input = X_tr.shape[1]
dimof_output = len(set(y_tr.flat))

y_tr = np_utils.to_categorical(y_tr.astype(int), dimof_output)

# test data
n_ts = 1000
X_ts1 = label(normal(n_ts, 1, sigma1, lambda1), l1)
X_ts2 = label(normal(n_ts, 1, sigma2, lambda2), l2)
X_ts = np.vstack((X_ts1, X_ts2))
X_ts, y_ts = shuffled(X_ts)
y_ts = np_utils.to_categorical(y_ts.astype(int), dimof_output)


model = Sequential([
        Dense(2, input_dim=1),
    ])

model.compile(loss='mse', optimizer='sgd')

# Train
model.fit(
    X_tr, y_tr, validation_split=0.2,
)


loss = model.evaluate(X_tr, y_tr)
print('metrics on training data')
print('loss: ', loss)
print()

loss = model.evaluate(X_ts, y_ts)
print('metrics on test data')
print('loss: ', loss)
print()
