import numpy as np
from tqdm import tqdm
from lib import gradient_descent

#A, b, x0 construction
N = 100

A = 10*np.eye(N)
A += -1*np.eye(N, k= 1)
A += -1*np.eye(N, k=-1)

b = np.zeros(N) + 8
b[0] = 9
b[-1] = 9

x0 = np.zeros(N)

y = gradient_descent(A, b, x0, its = 12)

print('Bound error', np.linalg.cond(A)*(np.sqrt(np.sum((A@y - b)**2)))/(np.sqrt(np.sum((b)**2))))
