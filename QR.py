import numpy as np
from numpy import linalg as la
import copy
def QR(A):
    m,n = A.shape
    R = copy.deepcopy(A)
    Q = np.eye(m)
    for k in range(n):
        u = copy.deepcopy(R[k:,k])
        u[0] = u[0] + 1.*np.sign(u[0])*la.norm(u)
        u = u/la.norm(u)
        u = u.reshape(-1,1)
        # print(np.dot(u,R[k:,k:]))
        H = np.eye(u.shape[0]) - 2*np.outer(u,u)
        R[k:,k:] = H.dot(R[k:,k:])
        # Q[k:,k:]= Q[k:,k:].dot(H)
        # R[k:,k:] -= 2.*np.dot(u,np.dot(R[k:,k:],u))
        Q[k:] = Q[k:] - 2*np.dot(u,np.dot(u.T,Q[k:]))

    return Q.T, R


A = np.array([[1.,2,3],[1,3,4], [5,2,3]])
Q,R = QR(A)
print(Q)
print("=====")
print(R)
print(np.dot(Q,R))
for i in range(Q.shape[1]):
    print(la.norm(Q[:,i]))
    print(Q[:,i].dot(Q[:,i-1]))
# print("========")
# Q1,R1 = la.qr(A)
# print("Numpy qr")
# print(Q1)
# print("=====")
# print(R1)
# print(np.dot(Q1,R1))
# print(np.isclose(Q,Q1,rtol = 0.001))
# print(np.isclose(R,R1, rtol = 0.001))