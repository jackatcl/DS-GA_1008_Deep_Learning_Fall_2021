import numpy as np


d = 5
n = 2 ** 5

X_train, X_test = np.random.uniform(-1, 1, (d, d, n)), np.random.uniform(-1, 1, (d, d, n))

def f_star(x):