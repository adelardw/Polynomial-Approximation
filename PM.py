import math
import numpy as np
import matplotlib.pyplot as plt

n = 200
# Data
xn = np.random.uniform(-10, 10, n)
yn = -np.sin(xn) / xn

m = 12  # polynomial exp -1 ( m - 1 )
right_eq = []  # vector in RHS
left_eq = []  # matrix in LHS

# vector in RHS
for k in range(0, m + 1):
    z = 0
    for i in range(0, n):
        z += yn[i] * (xn[i]) ** k
    right_eq.append(z)
right_eq = np.array(right_eq)

# matrix in LHS
for k in range(0, m + 1):
    g = []
    for j in range(0, m + 1):
        s = 0
        for i in range(0, n):
            s += (xn[i]) ** (j + k)
        g.append(s)
    left_eq.append(g)
left_eq = np.array(left_eq)


# Swapping string in LHS matrix. if diag elements ==0 -> swap
def Swapper_matr_full(A: list, b: list):
    p = len(A)
    for i in range(p):
        if A[i][i] == 0:
            for k in range(i + 1, p):
                if A[k][i] != 0:
                    A[i], A[k] = A[k], A[i]
                    b[i], b[k] = b[k], b[i]
    return A, b


# Gauss Method
def Gauss_method_func_full(A: list, b: list):
    p = len(A)
    A, b = Swapper_matr_full(A, b)
    for k in range(p - 1):
        for i in range(k + 1, p):
            if A[i][k] != 0 and A[k][k] != 0:
                global pre
                pre = (A[i][k] / A[k][k])
                b[i] -= pre * b[k]
                for j in range(k, p):
                    A[i][j] -= pre * A[k][j]
    return A, b


# inverse step
def backstep_full(A: list, b: list):
    p = len(A)
    xc = [0] * p
    xc[p - 1] = Gauss_method_func_full(A, b)[1][p - 1] / Gauss_method_func_full(A, b)[0][p - 1][p - 1]
    for k in range(1, p):
        s = 0
        for i in range(p - k, p):
            s = s + Gauss_method_func_full(A, b)[0][p - k - 1][i] * xc[i]
        xc[p - k - 1] = 1 / (Gauss_method_func_full(A, b)[0][p - k - 1][p - k - 1]) * (
                    Gauss_method_func_full(A, b)[1][p - k - 1] - s)
    return xc


solve1 = backstep_full(left_eq, right_eq)


def Approximate_Polynomial(A: list, b: list, x):
    construction = 0
    for i in range(m + 1):
        construction += backstep_full(A, b)[i] * x ** i
    return construction


P = [Approximate_Polynomial(left_eq, right_eq, i) for i in xn]
plt.figure(0)
# Numeric
plt.plot(xn, P, color="black")

# Analytic
plt.figure(1)
plt.plot(xn, yn, color="red")

# Point's
plt.figure(2)
plt.scatter(xn, yn, color="red", s=10)
plt.scatter(xn, P, color="black")
plt.show()
