import numpy as np
from tqdm import tqdm
from lib import gs

#A, b, x0 construction
N = 100

A = 10*np.eye(N)
A += -1*np.eye(N, k= 1)
A += -1*np.eye(N, k=-1)

b = np.zeros(N) + 8
b[0] = 9
b[-1] = 9

x0 = np.zeros(N) + 0.25
x0[0] = 0.5
x0[-1] = 0.5

omega = 2/(1 + np.sqrt(98/100))

y = gs(A, b, x0, its = 10, omega = omega, print_dx = True)

print('Bound error', np.linalg.cond(A)*(np.sqrt(np.sum((A@y - b)**2)))/(np.sqrt(np.sum((b)**2))))
