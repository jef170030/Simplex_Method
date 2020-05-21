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