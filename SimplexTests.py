import numpy as np
from Simplex1 import LinearProgram
import unittest


# (x,y,z) s.t. 1*x+2*y+ 1*z=1 has vertices (1,0,0),(0,1/2,0),(0,0,1)

class SimplexTests(unittest.TestCase):
    def test_example1(self):
        A = np.array([[1, 2, 1],
                      [0, 0, 1]])
        b = np.array([1, 0.5])
        c = np.array([1, 0.05, 0.1])

        lp1 = LinearProgram(c, A, b)

        x0 = np.array([0.5, 0, 0.5])
        indB0 = [0, 2]
        indN0 = [1]

        (status, x1, indB1, indN1) = lp1.SimplexStep(x0, indB0, indN0)
        # print(status)
        # print(x1)
        # print(indB1)
        # print(indN1)

        expected_status1 = LinearProgram.StepStatus.STEP_MADE
        expected_x1 = np.array([0, 0.25, 0.5])

        self.assertEqual(status, expected_status1)
        self.assertTrue(np.array_equal(x1, expected_x1))

        (status, x2, indB2, indN2) = lp1.SimplexStep(x1, indB1, indN1)

        # print(status)
        expected_status2 = LinearProgram.StepStatus.OPTIMAL_FOUND
        self.assertEqual(status, expected_status2)

        (status, x) = lp1.solve()
        self.assertTrue(np.array_equal(x, x2))
        self.assertEqual(status, LinearProgram.ProblemStatus.OPTIMAL_FOUND)

    def test_unbounded1(self):
        A = np.array([[1, 0, 0],
                      [0, 1, 0]])
        b = [1, 0.5]
        c = np.array([1, -0.5, -0.5])

        x0 = np.array([1, 0.5, 0])
        indB0 = [0, 1]
        indN0 = [2]

        lp2 = LinearProgram(c, A, b)
        (status, x1, indB1, indN1) = lp2.SimplexStep(x0, indB0, indN0)
        # print(status)
        # print(x1)
        # print(indB1)
        # print(indN1)

        expected_status = LinearProgram.StepStatus.UNBOUNDED
        self.assertEqual(status, expected_status)

    def test_auxproblem1(self):
        A = np.array([[1, 2]])
        b = np.array([1])
        c = np.array([1, 0.1])
        lp = LinearProgram(c, A, b)
        (lpaux, (x0aux, indBaux, indNaux)) = lp.getAuxiliaryProblem()

        (status, x, indB, indN) = lpaux.runSteps(x0aux, indBaux, indNaux)
        print(x)
        print(indB)
        print(indN)
        # Provided that the auxilary problem is non-degenerate, we must have indB as a subset of 1..n
        indNorig = [0]
        xorig = x[0: lp.n]
        print(xorig)
        (status, xorig, indBorig, indNorig) = lp.runSteps(xorig, indB, indNorig)
        print(xorig)
        print(status)


if __name__ == '__main__':
    unittest.main()
