import numpy as np
from enum import Enum


class LinearProgram:
    eps= 1e-10
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
        caux = np.concatenate((np.zeros_like(self.c), np.ones_like(self.b)))  # Cost auxiliary function

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
        q = indN[qN]  # where the minimizing argument qN of sN sits inside of N as an index
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
        p = indB[pB]  # Index of minimizer for v vector in B
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
        return status, x, indB, indN

    def phase_one_to_phase_two(self, xaux, indBaux):
        indB = []
        for i in range(len(indBaux)):
            if indBaux[i] < self.n:
                indB.append(indBaux[i])

        x = np.array(xaux[0: self.n])
        add_indices_amount = self.m -len(indB)

        #naive solution
        #TODO:account for a degenerate phase-2 problem
        i=0
        while add_indices_amount > 0:
            if (abs(x[i]) < self.eps) and (i not in indB):
                indB.append(i)
                add_indices_amount=add_indices_amount-1
            i=i+1

        indN = list(set(range(self.n)) - set(indB))
        return x, indB, indN

    def solve(self):
        (lpaux, (x0aux, indBaux, indNaux)) = self.getAuxiliaryProblem()
        (statusaux, xaux, indBaux, indNaux) = lpaux.runSteps(x0aux, indBaux, indNaux)

        if np.dot(lpaux.c, xaux) > 1e-9:
            return self.ProblemStatus.UNFEASIBLE, None

        (x, indB, indN) = self.phase_one_to_phase_two(xaux, indBaux)

        (status, x, indB, indN) = self.runSteps(x, indB, indN)

        if status == self.StepStatus.UNBOUNDED:
            return self.ProblemStatus.UNBOUNDED, None
        else:
            return self.ProblemStatus.OPTIMAL_FOUND, x
