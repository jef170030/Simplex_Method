import numpy as np
from Simplex1 import LinearProgram


class Network:
    def __init__(self, Q, cmin, cmax):
        self.Q = Q
        self.m = Q.shape[0]  # the amount vertices
        self.n = Q.shape[1]  # the amount of edges
        self.cmin = cmin
        self.cmax = cmax

    def getLinearProgram(self, w, l):
        c = np.concatenate((w, -w, np.zeros(2 * self.n)))
        A = np.block([
            [np.identity(self.n), -np.identity(self.n), np.identity(self.n), np.zeros((self.n, self.n))],
            [-np.identity(self.n), np.identity(self.n), np.zeros((self.n, self.n)), np.identity(self.n)],
            [self.Q, -self.Q, np.zeros((self.m, 2 * self.n))],
        ])
        b = np.concatenate((self.cmax, -self.cmin, l))
        return LinearProgram(c, A, b)
