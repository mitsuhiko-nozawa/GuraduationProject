import numpy as np

def L2norm(a):
    return np.sum(a**2)**0.5

def cossim(a1, a2):
    norm = L2norm(a1) * L2norm(a2)
    if norm == 0.0:
        return 0.0
    else:
        return np.dot(a1,a2) / norm
def mse(vec):
    vec = np.array(vec)
    return (np.sum(vec ** 2) / vec.shape[0])

def mae(vec):
    np.sum(np.abs(vec)) / vec.shape[0]



