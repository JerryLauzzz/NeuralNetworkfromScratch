import numpy as np


def bad_softmax(x):
    # numerical problem, potential blowup
    return np.exp(x) / np.sum(np.exp(x))


def softmax(x):
    # improve numerical problem
    x -= np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


a = [1.0, 2.0, 3.0]
print(bad_softmax(a), softmax(a))

b = [123, 456, 789]
print(bad_softmax(b), softmax(b))
