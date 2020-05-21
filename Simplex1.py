import numpy as np
class LinearProgram:
    def __init__(self, c, A, b):
        self.c = c
        self.A = A
        self.b = b
        self.m = A.shape[0]
        self.n = A.shape[1]

    def SimplexStep(self,x,indB,indN):



# Find matrices B and N using indB, indN, and A.
# Single step algorithm for simplex method

        newB = 0
        newN = 0
        return x, newB, newN


A1 = np.array([[1,1,1]])
b1 = 1
c1 = [1,2,3]

lp1 = LinearProgram(c1,A1,b1)

print(lp1.A)
print(A1.shape)