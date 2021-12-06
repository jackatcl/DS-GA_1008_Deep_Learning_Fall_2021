import numpy as np
import matplotlib.pyplot as plt

for d in [10, 100, 1000, 10000]:
    mean = np.zeros(d)
    cov = np.identity(d) / d
    x = np.random.multivariate_normal(mean, cov, 1000)
    x_norm_square = np.power(np.linalg.norm(x, axis=1), 2)
    print('d={}, mean={}, std={}'.format(d, np.mean(x_norm_square), np.std(x_norm_square)))
    plt.hist(x_norm_square, bins=100)
    plt.title('d={}'.format(d))
    plt.show()
