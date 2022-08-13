import numpy as np
from numpy import linalg as LA
from tqdm import tqdm
from lib import qr_method

N = 100

A = 10*np.eye(N)
A += -1*np.eye(N, k= 1)
A += -1*np.eye(N, k=-1)

P = np.identity(A.shape[0])

T, P = qr_method(A, P, T = 6)

print(np.diagonal(T))
print(P)
