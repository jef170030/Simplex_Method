import numpy as np
from Simplex1 import LinearProgram


class Network:
    def __init__(self, Q, cmin, cmax):
        self.Q = Q
        self.m = Q.shape[0]
        self.n = Q.shape[1]
        self.cmin = cmin
        self.cmax = cmax

    def getLinearProgram(self, w, l):
        c = np.concatenate((w, -w, np.zeros_like(self.n)), axis=None)
        A = np.block([
            [self.Q, -self.Q, np.zeros((self.m, self.n))],
            [np.identity(self.n), -np.identity(self.n), np.identity(self.n)],
            [-np.identity(self.n), np.identity(self.n), np.identity(self.n)]
        ])
        b = np.concatenate((l, self.cmax, self.cmin), axis=None)
        return LinearProgram(c, A, b)
