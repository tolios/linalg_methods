import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def error_bound(x, A, b):
    return np.linalg.cond(A)*(np.sqrt(np.sum((A@x-b)**2)))/(np.sqrt(np.sum((b)**2)))

def gs(A, b, x0, its = 15, omega = 1, plot = True, print_dx = False):
    N = A.shape[0]
    x_old = x0
    x_new = np.zeros(N)

    errors = []
    errors.append(error_bound(x_old, A, b))

    for it in tqdm(range(its), desc = 'Iterations'):
        for i in range(N):
            #update using old data
            s = 0
            for j in range(i):
                s -= A[i][j]*x_new[j]
            #update using new data
            w = 0
            for j in range(i+1, N):
                w -= A[i][j]*x_old[j]
            #relaxation method
            l = (1 - omega)*x_old[i]
            x_new[i] = l + (omega)*(w + s + b[i])/(A[i][i])

        if print_dx:
            print(f'it = {it}: ', np.sqrt(np.sum((x_new - x_old)**2)))
        x_old = x_new
        x_new = np.zeros(N)
        errors.append(error_bound(x_old, A, b))
    if plot:
        plt.title(f'Bound error plot for Gauss-Seidel (ω = {omega})')
        plt.ylabel('error')
        plt.xlabel('Iterations')
        plt.plot(errors)
        plt.show()
    return x_old

def jacobi(A, b, x0, its = 15, omega = 1, plot = True):
    N = A.shape[0]
    x_old = x0
    x_new = np.zeros(N)

    errors = []
    errors.append(error_bound(x_old, A, b))

    for _ in tqdm(range(its), desc = 'Iterations'):
        for i in range(N):
            s = 0
            for j in range(N):
                if j != i:
                    s -= A[i][j]*x_old[j]
            #relaxation method
            l = (1 - omega)*x_old[i]
            x_new[i] = l + (omega)*(s + b[i])/(A[i][i])

        x_old = x_new
        x_new = np.zeros(N)
        errors.append(error_bound(x_old, A, b))
    if plot:
        plt.title(f'Bound error plot for Jacobi (ω = {omega})')
        plt.ylabel('error')
        plt.xlabel('Iterations')
        plt.plot(errors)
        plt.show()
    return x_old

def gradient_descent(A, b, x0, its = 12, plot = True):

    N = A.shape[0]
    x_old = x0
    x_new = np.zeros(N)

    errors = []
    errors.append(error_bound(x_old, A, b))

    for it in tqdm(range(its), desc = 'Iterations'):

        r = b - (A@x_old)

        a = np.sum(r*r)/np.sum(r*(A@r))

        x_new = x_old + a*r
        x_old = x_new
        x_new = np.zeros(N)
        errors.append(error_bound(x_old, A, b))
    if plot:
        plt.title(f'Bound error plot for Gradient Descent')
        plt.ylabel('error')
        plt.xlabel('Iterations')
        plt.plot(errors)
        plt.show()

    return x_old

def conjugate_descent(A, b, x0, its = 12, plot = True):

    N = A.shape[0]
    x_old = x0
    x_new = np.zeros(N)

    errors = []
    errors.append(error_bound(x_old, A, b))

    for it in tqdm(range(its), desc = 'Iterations'):

        if it == 0:
            r = b - (A@x_old)
            p = b - (A@x_old)

        a = np.sum(p*r)/np.sum(p*(A@p))

        x_new = x_old + a*p

        r = r - a*(A@p)

        beta  = np.sum(((A@p).T)*r)/np.sum(((A@p).T)*p)

        p = r - beta*p

        x_old = x_new
        x_new = np.zeros(N)
        errors.append(error_bound(x_old, A, b))

    if plot:
        plt.title(f'Bound error plot for Conjugate Descent')
        plt.ylabel('error')
        plt.xlabel('Iterations')
        plt.plot(errors)
        plt.show()

    return x_old

def R(x, A):

    return (x.T@A@x)/(x.T@x)

def power_method(A, x0, T = 5):

    x = x0

    for t in range(T):

        y = A@x
        x = y/(np.sqrt(y.T@y))

    return x, R(x, A)

def qr_method(A, P, T = 6):

    for t in range(T):

        q, r = np.linalg.qr(A)
        A = r@q
        P = P@q

    return A, P
