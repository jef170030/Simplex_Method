import numpy as np
from Networks import Network

Q = np.array([[1, 0, 1, 0, 0, -1],
              [-1, 1, 0, 1, 0, 0],
              [0, 0, -1, -1, 1, 0],
              [0, -1, 0, 0, -1, 1]])

cmin = np.array([-3, -5, -1, 0, 0, -100])
cmax = np.array([1, 2, 3, 3, .5, 100])

net1 = Network(Q, cmin, cmax)

w = np.array([0, 0, 0, 0, 0, -1])
l = np.array([0, 0, 0, 0])

net1lp = net1.getLinearProgram(w, l)
(lpaux, (x0aux, indBaux, indNaux)) = net1lp.getAuxiliaryProblem()


print("Net1.m", net1.m)
print("Net1.n", net1.n)

print("net1lp.m", net1lp.m)   #constraints:  16=6+6+4
print("net1lp.n", net1lp.n)   #variables:    24=6+6+2*6

print("lpaux.m", lpaux.m)     #constraints
print("lpaux.n", lpaux.n)     #variables: 40 = 24+16

(statusaux, xaux, indBaux, indNaux) = lpaux.runSteps(x0aux, indBaux, indNaux)
(x, indB, indN) = net1lp.phase_one_to_phase_two(xaux, indBaux)
print(statusaux)
print("xaux=")
print(xaux)
print("x_lp=")
print(xaux[range(24)])
print(x)
print("indBaux=")
print(indBaux)
print(len(indBaux))
print("indB=")
print(indB)
print("indN=")
print(indN)
print("flow=")
print(net1.lp_point_to_flow(xaux))

(status,x) = net1lp.solve()
print(status)
print("x=")
print(x)