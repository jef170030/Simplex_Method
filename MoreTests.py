import numpy as np
import unittest
from SimplexTests import SimplexTests
from Simplex1 import LinearProgram


A = np.array([[1, 1, 1, 1, 1]])
b = np.array([1])
c = np.array([0, 0, 0, 0, -1])

lp5 = LinearProgram(c, A, b)

(status, x) = lp5.solve()
print("5D problem:")
print(status)
print(x)


A = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
b = np.array([1, 0.5])
c = np.array([1, 0, 0.5, 0, 0, 0, 0, 0, 0, -0.5])

lp10 = LinearProgram(c, A, b)

(status, x) = lp10.solve()
print("10D problem:")
print(status)
print(x)


"""Network test
A = np.array([[1, 0, 1, 0, 0, -1],
              [-1, 1, 0, 1, 0, 0],
              [0, 0, -1, -1, 1, 0],
              [0, -1, 0, 0, -1, 1]])
b = np.array([0, 0, 0, 0])
c = np.array([0, 0, 0, 0, 0, -1])

NetTest = LinearProgram(c, A, b)

(status, x) = NetTest.solve()
print(status)
print(x)
"""

