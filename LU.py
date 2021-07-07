import copy
import random
import datetime

def backsolve(A,b):
    x = [0 for i in b]
    n = len(b)
    for i in range(n-1, -1, -1):
        s = b[i]
        for j in range(i, n):
            s-=A[i][j]*x[j]
        x[i] = s/A[i][i]
    return x



def gauss(A,b):
    n = len(A)
    # rows
    # print(type(b))
    try:
        if type(b[0]) !=list:
            for k in range(n-1):
                for i in range(k+1,n):
                    piv = A[i][k]/A[k][k]
                    for j in range(k,n):
                        A[i][j] -= piv*A[k][j]
                    b[i] -= piv*b[k]
            x = backsolve(A,b)
        else:
            for k in range(n-1):
                for i in range(k+1,n):
                    piv = A[i][k]/A[k][k]
                    for j in range(k,n):
                        A[i][j] -= piv*A[k][j]
                    for l in range(n):
                        b[i][l] -= piv*b[k][l]
            x = []
            for i in range(n):
                x.append(backsolve(A,[b[j][i]for j in range(n)]))
        return x

    except ZeroDivisionError as err:
        print("Matrix singular")
    

   

def eye(n):
    return [[1 if i == j else 0 for j in range(n) ]for i in range(n)]

def Inverse(A):
    n = len(A)
    return gauss(copy.deepcopy(A),eye(n))


n = 3
A = [[random.randint(1,3) for j in range(n)] for i in range(n)]

tic = datetime.datetime.now()
A_inv1= Inverse(A)
print(A_inv1)
toc = datetime.datetime.now()
print(toc-tic)

