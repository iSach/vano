import numpy as np
from functools import partial

# -----------------------------------------------------------------------------
#                     Generalized Maximum Mean Discrepancy
# -----------------------------------------------------------------------------

# Naive implementation
# TODO: * vectorize
#         -- https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html#references
#            contains implementation with numpy broadcasting
#         -- solution: make a matrix of k[i,j] = k(X[i], Y[j])
# TODO: * Try with 2D Gaussian densities output from the model.
def kernel_family(sigma_min=0.1, sigma_max=20.0):
    """Kernel family for the generalized maximum mean discrepancy (MMD).

    Parameters
    ----------
    sigma_min : float, optional
        Minimum bandwidth (1st kernel)
    sigma_max : float, optional
        Maximum bandwidth (last kernel)

    Returns
    -------
    kernels : List of callables
        Each callable takes two arguments, `X` and `Y`, and returns the
        kernel matrix of shape `(X.shape[0], Y.shape[0])`.

    """
    def kernel_gaussian(x, y, sigma):
        z = x - y
        squared_norm = np.linalg.norm(z)**2
        k = squared_norm / (2 * sigma**2)
        return np.exp(-k)
    
    def kernel_multiscale(x, y, sigma):
        pass # todo

    return [partial(kernel_gaussian, sigma=sigma) for sigma in np.linspace(sigma_min, sigma_max, 50)]

def mmd(X, Y, k):
    N, M = len(X), len(Y)
    rN, rM = range(N), range(M)

    # Paper, formulated as:
    # sum_{i}^{N} sum_{j}^{N} k(x_i, x_j)
    #MMD_xx = 1 / N**2 * np.sum([k(X[i], X[j]) for i in rN for j in rN])
    #MMD_yy = 1 / M**2 * np.sum([k(Y[i], Y[j]) for i in rM for j in rM])
    #MMD_xy = 1 / (N * M) * np.sum([k(X[i], Y[j]) for i in rN for j in rM])

    # https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html#references
    # Formal Definition 3.: j ≠ i

    MMD_xx = 1 / (N * (N-1)) * np.sum([k(X[i], X[j]) if i != j else 0.0 for i in rN for j in rN])
    MMD_yy = 1 / (M * (M-1)) * np.sum([k(Y[i], Y[j]) if i != j else 0.0 for i in rM for j in rM])
    MMD_xy = 1 / (N * M) * np.sum([k(X[i], Y[j]) for i in rN for j in rM])

    return MMD_xx + MMD_yy - 2 * MMD_xy

def gmmd(X, Y):
    """Generalized maximum mean discrepancy (MMD).

    Parameters
    ----------
    X : array_like
        First sample, shape `(n_samples, n_features)`
    Y : array_like
        Second sample, shape `(n_samples, n_features)`

    Returns
    -------
    gmmd : float
        Generalized MMD (sup of MMDs over a family of kernels)
    mmds : List of floats
        MMDs for each kernel in the family

    """
    kernels = kernel_family()
    mmds = [mmd(X, Y, k) for k in kernels]
    gmmd = np.max(mmds)

    return gmmd, mmds

# -- test --

import matplotlib.pyplot as plt

X = np.random.normal(0, 1, size=(100, 2))
Y = np.random.normal(1, 1, size=(100, 2))

gmmd, mmds = gmmd(X, Y)


sigmas = np.linspace(0.1, 20.0, 50)

print(mmds)
print('max', np.max(mmds))
print('min', np.min(mmds))
print('sigma_max', sigmas[np.argmax(mmds)])
print('sigma_min', sigmas[np.argmin(mmds)])

plt.plot(sigmas, mmds)
plt.xlabel(r"$\sigma$")
plt.ylabel("MMD")
plt.axhline(gmmd, color="red", linestyle="--")
plt.axvline(sigmas[np.argmax(mmds)], color="red", linestyle="--")
plt.title("Generalized Maximum Mean Discrepancy")
plt.show()
