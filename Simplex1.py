import numpy as np
class LinearProgram:
    def __init__(self, c, A, b):
        self.c = c
        self.A = A
        self.b = b
        self.m = A.shape[0]
        self.n = A.shape[1]

    def SimplexStep(self,x,indB,indN):
        B=self.A[:,indB]
        N=self.A[:,indN]
        cB=self.c[indB]
        cN=self.c[indN]
        print(B)
        print(N)
        print(cB)
        print(cN)

        l=np.linalg.solve(np.transpose(B),cB) #find lambda as lambda=B^(-T)c_B
        print(l)
        sN=cN-np.matmul(np.transpose(N),l)
        print(sN)
        q=np.argmin(sN) #naive selection of q
        print(q)
        if sN[q]>=0:
            print("We are done")
            return None
            # TODO:implement stop on all components of sN>=0 - i.e. we are optimal
        print(A[:,q])
        d=np.linalg.solve(B,A[:,q])
        print(d)
        #we stopped here
        newB = 0
        newN = 0
        return x, newB, newN

#(x,y,z) s.t. 1*x+2*y+ 1*z=1 has vertices (1,0,0),(0,1/2,0),(0,0,1)


A = np.array([[1,2,1],
              [0,0,1]])
b = [1,0.5]
c = np.array([1,0.05,0.1])

x0=[0.5,0,0.5]
indB0=[0,2]
indN0=[1]

lp1 = LinearProgram(c,A,b)
lp1.SimplexStep(x0,indB0,indN0)
