import numpy as np
from Networks import Network

Q = np.array([[1, 0, 1, 0, 0, -1],
              [-1, 1, 0, 1, 0, 0],
              [0, 0, -1, -1, 1, 0],
              [0, -1, 0, 0, -1, 1]])
cmin = np.array([0, 0, 0, 0, 0, 0])
cmax = np.array([1, 2, 3, 3, .5, np.inf])

Net1 = Network(Q, cmin, cmax)

w = np.array([0, 0, 0, 0, 0, -1])
l = np.array([0, 0, 0, 0])

net1lp = Net1.getLinearProgram(w, l)
(lpaux, (x0aux, indBaux, indNaux)) = net1lp.getAuxiliaryProblem()
(statusaux, xaux, indBaux, indNaux) = lpaux.runSteps(x0aux, indBaux, indNaux)

# (status,x) = net1lp.solve()
print(statusaux)
print(xaux)
