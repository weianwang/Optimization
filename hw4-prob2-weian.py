import numpy as np
from scipy import linalg, optimize
import pdb
from fractions import Fraction


def make_identity(n):
    """
    Simple helper function to quickly make an identity matrix.
    Parameters:
        -n: dimensions of square matrix
    Returns identity matrix.
    """
    return np.matrix(np.identity(n), copy=False)


def newton_inverse(A, n, eps):
    """
    Implementation of Newton's method to compute the inverse of a matrix
    Parameters/Inputs:
        -A: matrix in R^{nxn} that is invertible
        -n: dimensions of A
        -eps: stopping condition
    Returns approximation of matrix inverse
    """
    alpha_max = 2 / (linalg.norm(A, 'fro') ** 2)
    #arbitrary alpha selected from range
    alpha = np.random.random()*alpha_max
    #Initialize X0 as alpha * A^T
    X0 = alpha * np.transpose(A)

    I = make_identity(n)
    sqrtn = np.sqrt(n)
    X_old = X0
    X_new = np.dot(X_old, (2 * I - np.dot(A, X_old)))
    #Compute approximation of Frobenius norm of difference
    #    to determine stopping condition.
    D = X_new - X_old
    test_norm = sqrtn * linalg.norm(D, 2)

    while (test_norm > eps):

        X_old = X_new
        X_new = np.dot(X_old, (2 * I - np.dot(A, X_old)))
        D = X_new - X_old
        # Recalculate to test for stopping condition.
        test_norm =  sqrtn * linalg.norm(D, 2)

    return X_new


#2(d)(i)
#2x2 matrix of integer values
A = np.random.randint(-2000, 2000, size=(2, 2))
# known inverse based on build-in function
A_inv = linalg.inv(A)
# approximated inverse using Newton method
newton_A_inv = newton_inverse(A, 2, 0.00001)
fwd_err = linalg.norm(newton_A_inv - A_inv, 'fro')

print(A)
print(A_inv)
print(newton_A_inv)
print(fwd_err)

#10x10 Diagonal entries with rational entries on the diagonal
A = np.zeros([10, 10])
#No significance to 2000. Arbitrary selection of large integer
num = np.random.randint(-2000, 2000, size=(10, 1))
#If any zeros in numerator, initialize the array again.
while (np.any(num == 0)):
    num = np.random.randint(-2000, 2000, size=(10, 1))

denom = np.random.randint(1, 2000, size=(10, 1))

denom = denom.astype(float)

diag = num * (1 / denom)
for i in range(0, 10):
    A[i, i] = diag[i]

A_inv = linalg.inv(A)
# approximated inverse using Newton method
newton_A_inv = newton_inverse(A, 10, 0.00001)
fwd_err = linalg.norm(newton_A_inv - A_inv, 'fro')

print(A)
print(A_inv)
print(newton_A_inv)
print(fwd_err)

#2(d)(ii)
#Testing with dimensions of matrix: 10, 100, 1000, 10000
#n = 10
I_10 = make_identity(10)
A_10 = np.random.randn(10, 10)
Y_10_star = linalg.inv(A_10)
X_10_star = newton_inverse(A_10, 10, 0.00001)
err_Y_star = linalg.norm(I_10 - np.dot(A_10, Y_10_star), 'fro')
err_X_star = linalg.norm(I_10 - np.dot(A_10, X_10_star), 'fro')

#print(err_Y_star)
#print(err_X_star)

#n = 100
I_100 = make_identity(100)
A_100 = np.random.randn(100, 100)
Y_100_star = linalg.inv(A_100)
X_100_star = newton_inverse(A_100, 100, 0.00001)
err_Y_star = linalg.norm(I_100 - np.dot(A_100, Y_100_star), 'fro')
err_X_star = linalg.norm(I_100 - np.dot(A_100, X_100_star), 'fro')

print(err_Y_star)
print(err_X_star)


#n = 1000
I_1000 = make_identity(1000)
A_1000 = np.random.randn(1000, 1000)
Y_1000_star = linalg.inv(A_1000)
X_1000_star = newton_inverse(A_1000, 1000, 0.00001)
err_Y_star = linalg.norm(I_1000 - np.dot(A_1000, Y_1000_star), 'fro')
err_X_star = linalg.norm(I_1000 - np.dot(A_1000, X_1000_star), 'fro')

print(err_Y_star)
print(err_X_star)



