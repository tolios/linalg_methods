import numpy as np
from numpy import linalg as LA
from tqdm import tqdm
from lib import R, power_method

N = 100

A = 10*np.eye(N)
A += -1*np.eye(N, k= 1)
A += -1*np.eye(N, k=-1)

x = np.ones(N)

u, l = power_method(A, x, T = 3)

print(l)
print(u)
