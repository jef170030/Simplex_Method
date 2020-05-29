import numpy as np
from enum import Enum


class LinearProgram:
    class StepStatus(Enum):
        STEP_MADE=0
        OPTIMAL_FOUND=1
        UNBOUNDED=2

    def __init__(self, c, A, b):
        self.c = c
        self.A = A
        self.b = b
        self.m = A.shape[0]
        self.n = A.shape[1]

    def SimplexStep(self, x, indB, indN):
        B = self.A[:, indB]
        N = self.A[:, indN]
        cB = self.c[indB]
        cN = self.c[indN]
        xB = x[indB]

        l = np.linalg.solve(np.transpose(B), cB)  # Find lambda as lambda = B^(-T)c_B

        sN = cN - np.matmul(np.transpose(N), l)
        qN = np.argmin(sN)  # naive selection of q
        q=indN[qN]
        if sN[qN] >= 0:
            return self.StepStatus.OPTIMAL_FOUND, x, indB, indN

        d = np.linalg.solve(B, self.A[:, q])

        if all(di <= 0 for di in d):
            return self.StepStatus.UNBOUNDED, None, None, None

        v=np.zeros_like(d)
        for i in range(len(d)):
            if d[i] > 0:
                v[i]=xB[i]/d[i]
            else:
                v[i]=np.Inf
        pB=np.argmin(v)
        p=indB[pB]
        xqplus=v[pB]

        xBplus=xB-np.multiply(xqplus,d)
        xNplus=np.zeros(len(indN))
        xNplus[qN]=xqplus

        xplus=np.zeros_like(x)
        xplus[indB]=xBplus
        xplus[indN]=xNplus

        indBplus=indB[:]
        indNplus=indN[:]
        indBplus.append(q)
        indNplus.append(p)

        indBplus.pop(pB)
        indNplus.pop(qN)
        return self.StepStatus.STEP_MADE,xplus,indBplus,indNplus
