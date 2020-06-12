import numpy as np
from Simplex1 import LinearProgram


class Network:
    def __init__(self, Q, cmin, cmax):
        self.Q = Q
        self.cmin = cmin
        self.cmax = cmax

    def getLinearProgram(self, w, l):
        c = w
        A = self.Q
        b = l
        return LinearProgram(c, A, b)
