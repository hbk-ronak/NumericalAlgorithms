import numpy as np
from numpy import linalg as la
TOL = 1e-8

def arnoldi(A,b,m):
    n = A.shape[0]
    H = np.zeros((m+1,m))
    V = np.zeros((n,m))
    V[:,0] = b.reshape(-1,)/la.norm(b)
    for j in range(m):
        w = np.dot(A,V[:,j])
        for i in range(j+1):
            H[i,j] = np.dot(w,V[:,i])
            w-=H[i,j]*V[:,i]
        H[j+1,j] = la.norm(w)
        if H[j+1,j] <= TOL:
            break
        V[:,j+1]=w/H[j+1,j]
    return V,H,j

if __name__ ==  "__main__":
    A = np.array([[1,2],[3,1]])
    b = np.array([1,2])
    v,h,j = arnoldi(A,b,2)
    print(v)

    print("=======")

    print(h)

    print(j)