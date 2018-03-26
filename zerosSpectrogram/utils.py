import numpy as np


def extr2minth(M,th):

    C,R = M.shape

    Mid_Mid = np.zeros((C,R), dtype=bool)

    for c in range(1, C-1):
        for r in range(1, R-1):
            T = M[c-1:c+2,r-1:r+2]
            Mid_Mid[c, r] = (np.min(T) == T[1, 1]) * (np.min(T) > th)
            #Mid_Mid[c, r] = (np.min(T) == T[1, 1])

    x, y = np.where(Mid_Mid)
    return x, y
