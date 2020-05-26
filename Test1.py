import numpy as np
def Test(x):
    return np.linalg.inv(x)

A=np.array([[2,1],[1,2]])
b=np.array([np.e, np.pi])

x=Test(A)

#print(b)
#print(x)

#print(np.dot(x,b))
#print(np.matmul(x,A))

C=np.array([[2,1],[1,2],[9,9]])
print(C.shape[0])

A=np.array([[1,  2, 3,-1,-2],
            [4,  5, 6,-3,4],
            [7,  8, 9,-5,-6],
            [10,11,12,-7,-8]])
ind=[0,4,1]
print(A[:,ind])

