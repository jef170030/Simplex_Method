import numpy as np
from enum import Enum


class LinearProgram:
    class StepStatus(Enum):
        STEP_MADE = 0
        OPTIMAL_FOUND = 1
        UNBOUNDED = 2

    class ProblemStatus(Enum):
        OPTIMAL_FOUND = 0
        UNBOUNDED = 1
        UNFEASIBLE = 2

    def __init__(self, c, A, b):
        self.c = c
        self.A = A
        self.b = b
        self.m = A.shape[0]
        self.n = A.shape[1]

    def getAuxiliaryProblem(self):
        sign = np.where(self.b >= 0, 1, -1)
        E = np.diag(sign)
        Aaux = np.concatenate((self.A, E), axis=1)
        caux = np.concatenate((np.zeros_like(self.c), np.ones_like(self.b)))

        x0aux = np.concatenate((np.zeros_like(self.c), np.absolute(self.b)))
        indBaux = list(range(self.n, self.n + self.m))
        indNaux = list(range(0, self.n))
        return LinearProgram(caux, Aaux, self.b[:]), (x0aux, indBaux, indNaux)

    def SimplexStep(self, x, indB, indN):
        B = self.A[:, indB]
        N = self.A[:, indN]
        cB = self.c[indB]
        cN = self.c[indN]
        xB = x[indB]

        l = np.linalg.solve(np.transpose(B), cB)  # Find lambda as lambda = B^(-T)c_B

        sN = cN - np.matmul(np.transpose(N), l)
        qN = np.argmin(sN)  # naive selection of q
        q = indN[qN]
        if sN[qN] >= 0:
            return self.StepStatus.OPTIMAL_FOUND, x, indB, indN

        d = np.linalg.solve(B, self.A[:, q])

        if all(di <= 0 for di in d):
            return self.StepStatus.UNBOUNDED, None, None, None

        v = np.zeros_like(d)
        for i in range(len(d)):
            if d[i] > 0:
                v[i] = xB[i] / d[i]
            else:
                v[i] = np.Inf
        pB = np.argmin(v)
        p = indB[pB]
        xqplus = v[pB]

        xBplus = xB - np.multiply(xqplus, d)
        xNplus = np.zeros(len(indN))
        xNplus[qN] = xqplus

        xplus = np.zeros_like(x)
        xplus[indB] = xBplus
        xplus[indN] = xNplus

        indBplus = indB[:]
        indNplus = indN[:]
        indBplus.append(q)
        indNplus.append(p)

        indBplus.pop(pB)
        indNplus.pop(qN)
        return self.StepStatus.STEP_MADE, xplus, indBplus, indNplus

    def runSteps(self, x, indB, indN):
        status = self.StepStatus.STEP_MADE
        i = 0
        while status == self.StepStatus.STEP_MADE:
            (status, x, indB, indN) = self.SimplexStep(x, indB, indN)
            i = i + 1
        return (status, x, indB, indN)

    def solve(self):
        (lpaux, (x0aux, indBaux, indNaux)) = self.getAuxiliaryProblem()
        (statusaux, xaux, indBaux, indNaux) = lpaux.runSteps(x0aux, indBaux, indNaux)

        if np.dot(lpaux.c, xaux) > 1e-9:
            return self.ProblemStatus.UNFEASIBLE, None
        # Provided that the auxilary problem is non-degenerate, we must have indB as a subset of 1..n
        # TODO: check if all elements of indB are less then n
        # TODO: implement the case of degenerate problem

        x = np.array(xaux[0: self.n])
        indB = indBaux
        indN = []
        for i in range(0, self.n):
            if not (i in indB):
                indN.append(i)

        (status, x, indB, indN) = self.runSteps(x, indB, indN)

        if status == self.StepStatus.UNBOUNDED:
            return self.ProblemStatus.UNBOUNDED, None
        else:
            return self.ProblemStatus.OPTIMAL_FOUND, x
