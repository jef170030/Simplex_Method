import numpy as np
from Networks import Network
from Simplex1 import LinearProgram

Q = np.array([[1, 0, 1, 0, 0, -1],
              [-1, 1, 0, 1, 0, 0],
              [0, 0, -1, -1, 1, 0],
              [0, -1, 0, 0, -1, 1]])
cmin = np.array([0, 0, 0, 0, 0, 0])
cmax = np.array([1, 2, 3, 3, .5, np.inf])

Net1 = Network(Q, cmin, cmax)

w = np.array([0, 0, 0, 0, 0, -1])
l = np.array([0, 0, 0, 0])

(status, x) = Net1.getLinearProgram(w, l)
print(status)
print(x)
