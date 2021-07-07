import numpy as np
from numpy import linalg as la
import FOM as f
TOL = 1e-8
def IOM(A,b,m,k):
    n = A.shape[0]
    H = np.zeros((m+1,m))
    V = np.zeros((n,m+1))
    V[:,0] = b.reshape(-1,)/la.norm(b)
    for j in range(m):
        w = np.dot(A,V[:,j])
        for i in range(max(0,j-k+1),j+1):
            H[i,j] = np.dot(w,V[:,i])
            w-=H[i,j]*V[:,i]
        H[j+1,j] = la.norm(w)
        if H[j+1,j] <= TOL:
            break
        V[:,j+1]=w/H[j+1,j]
    return V,H,j

def IOMSolve(A,b,x0,m,k):
    r0 = b-np.dot(A,x0)
    beta = la.norm(r0)
    V,H,j = IOM(A,r0,m,k)
    Hm = H[:j+1,:j+1]
    HmInv = la.solve(Hm, np.eye(j+1))
    ym = np.dot(HmInv, beta*np.eye(j+1,1))
    xm = x0 + np.dot(V[:j+1,:j+1],ym)
    return xm

if __name__ == "__main__":
    A = np.random.randint(low = 1, high = 5, size = (5,5))
    b = np.random.randint(low = 1, high = 5, size = (5,1))
    x0 = np.ones(shape = (5,1))
    m = 10
    k = 4
    print("Results from Direct method", la.solve(A,b))

    print("=======")

    xm = IOMSolve(A,b,x0,m,k)
    V,H,j = IOM(A,b,m,k)
    print("Results from IOM iteration: ", xm)

    print("=======")

    print("Results from FOM ", f.FOM(A,b,x0,m))