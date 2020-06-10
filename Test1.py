import numpy as np
from Simplex1 import LinearProgram

def Test(x):
    return np.linalg.inv(x)

A=np.array([[2,1],[1,2]])
b=np.array([np.e, np.pi])

x=Test(A)

#print(b)
#print(x)

#print(np.dot(x,b))
#print(np.matmul(x,A))

C = np.array([[2,1],[1,2],[9,9]])
print(C.shape[0])

A = np.array([[1,  2, 3,-1,-2],
            [4,  5, 6,-3,4],
            [7,  8, 9,-5,-6],
            [10,11,12,-7,-8]])


def function_arg_test(mylist):
    mylistlocal = mylist[:]
    mylistlocal.append(-1)
    return mylistlocal

ind = [0,4,1]
ind2 = function_arg_test(ind)
print(ind)


u = np.array([1,2,3,4])
v = np.diag(u)
print(v)

u = np.array([-2,0,10])
print(np.sign(u))

A = np.array([[1, 2, 1],
              [0, 0, 1],
              [1, 0, 0]])
b = np.array([2, 0 ,-10,])
c = np.array([1, 0.05, 0.1])

lp1 = LinearProgram(c, A, b)
print(np.diag([1]))

n = 5
m = 3

indBaux = list(range(n, n + m))
indNaux = list(range(0, n))
print(indBaux)
print(indNaux)