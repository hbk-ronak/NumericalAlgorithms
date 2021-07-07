import numpy as np
from numpy import linalg as la
import arnoldi as a
TOL = 1e-8

def FOM(A,b,x0,m):
    r0 = b-np.dot(A,x0)
    beta = la.norm(r0)
    V,H,j = a.arnoldi(A,r0,m)
    Hm = H[:j+1,:j+1]
    HmInv = la.solve(Hm, np.eye(j+1))
    ym = np.dot(HmInv, beta*np.eye(j+1,1))
    xm = x0 + np.dot(V[:j+1,:j+1],ym)
    return xm


if __name__ == "__main__":
    A = np.array([[1,2,3],[1,2,4], [5,2,3]])
    b = np.array([1,2,4]).reshape(-1,1)
    xm = FOM(A,b,x0 = np.array([1,1,1]).reshape(-1,1), m = 4)

    print("=======")
    print("Results from FOM Arnoldi iteration: ", xm)

    print("=======")

    print("Results from Direct method", la.solve(A,b))
    # print(np.dot(A,b).shape)