import numpy as np
from scipy import linalg, optimize
import pdb
import matplotlib.pyplot as plt #source: matplotlib.org

#################### DEFINITIONS ####################


def func(A, b, c_vec, x):
    """
    Implementation of the specified objective function, computed at value x.

    Parameters:
        -A is R^{500x100} matrix
        -b is vector in R^{500}
        -c_vec is vector in R^{100},
        -x is vector in R^{100}
    Returns value of function evaluated at x (real number).
    """
    dp = np.dot(np.transpose(c_vec), x)
    return dp - np.sum(np.log(b - np.dot(A, x)))


def dfunc(A, b, c_vec, x):
    """
    Implementation of gradient of specified objective function.
    Gradient computed by hand in attached work.
    Parameters:
        -A is R^{500x100} matrix
        -b is vector in R^{500}
        -c_vec is vector in R^{100},
        -x: vector in R^{100} where we want the gradient evaluated
    Returns grad: vector in R^{100} of gradient evaluated at vector x
    """

    #Computation of the denominator in the
    #    summation term of the partial derivatives.
    #We store this value in an array of length 500.
    grad = np.empty([100, 1])
    sum_denom = np.empty([500, 1])

    for i in range(0, 500):
        A_row = np.transpose(A[i, :])
        dp = np.dot(A_row, x)
        sum_denom[i] = np.reciprocal(b[i] - dp)

    #Iterate over all k from 1 to 100.
    for k in range(0, 100):
        numer = np.transpose(A[:, k])
        sum_term = np.dot(numer, sum_denom)
        grad[k] = c_vec[k] + sum_term

    return grad


def hess_func(A, b, c_vec, x):
    """
    Implementation of approximation of hessian computed at a specific x.
    Computation of second partial derivatives can be found in the attached work.
    Implementation involves nested for loops.
    We exploit assumed symmetry of the Hessian to reduce computations.
    Parameters:
        -A is R^{500x100} matrix
        -b is vector in R^{500}
        -c_vec is vector in R^{100},
        -x: vector in R^{100}
    Returns 100x100 matrix with real values.
    """

    hess = np.empty([100, 100])

    #Computation of the denominator in the
    #    summation term of the partial derivatives.
    #We store this value in an array of length 500.

    sum_denom = np.empty([500, 1])

    for i in range(0, 500):
        A_row = A[i, :]
        dp = np.vdot(A_row, x)
        denom = (b[i] - dp) ** 2
        sum_denom[i] = np.reciprocal(denom)

    for k in range(0, 100):
        A_row_k = A[:, k]
        for j in range(k, 100):
            A_row_j = A[:, j]
            numer_prod = A_row_k * A_row_j

            #Second order partial derivative
            par2 = np.vdot(numer_prod, sum_denom)
            hess[k, j] = hess[j, k] = par2

    return hess

#################### IMPLEMENTATION OF ALGORITHMS ####################


def backtrack_line_search(f, df, A, b, c, rho, xk, pk):
    """
    Implementation of Backtracking Line Search.
    Parameters:
        - f: function that takes a vector in R^{100}
        - df: gradient of function f, takes vector in R^{100} as input
        - A: 500x100 matrix, one parameter in specified f
        - b: vector in R^{500}, one parameter in specified f
        - c, rho: parameters to check satisfying Armijo conditions
        - xk: vector in R^{100} as input
        - pk: descent direction, vector in R^{100}
    Returns step size, denoted alpha.
    """

    #Initialize alpha to 1
    alpha = 1

    #Compute f and df at xk
    fxk = f(xk)
    der = df(xk)

    xk_new = xk + alpha * pk

    #Check based on value of fx if xk is in feasible range
    #If outside of domain, decrease alpha by rho
    #Such an alpha exists given f is a continuous function.
    while np.any(b - np.dot(A, xk_new) <= 0):

        alpha = alpha * rho
        #Update xk because we're out of the domain
        xk_new = xk + alpha * pk

    #Now that xk is within the domain,
    #    loop until the Armijo condition is satisfied.
    armijo_cond = fxk + c * alpha * np.dot(np.transpose(der), pk)
    fx_new = f(xk_new)
    while fx_new > armijo_cond:
        #Reduce alpha (step size)
        alpha = alpha * rho
        xk_new = xk + alpha * pk
        fx_new = f(xk_new)
        armijo_cond = fxk + c * alpha * np.dot(np.transpose(der), pk)

    return alpha


def steepest_descent(fn, dfn, A, b, c_vec, c, rho, x_0, eps):
    """
    Implementation of steepest descent algorithm.
    Follows pseudocode presented in lecture.
    Parameters:
        -fn: function with vector in R^{100}, takes parameters A,b, and c_vec
        -dfn: gradient of function f, takes vector in R^{100} as input
                and paramters A, b, and c_vec
        -A: 500x100 real-valued matrix
        -b: vector in R^{500}
        -c_vec: vector in R^{100}
        -x0: vector in R^{100} taken as input
        -eps: tolerance level
    Returns list featuring:
        -list[0]: xk, which is the approximation of the minimizer
        -list[1]: err_arr, which is the array of errors in the approximation
        -list[2]: counter, which is number of iterations
    """

    f = lambda x: fn(A, b, c_vec, x)
    df = lambda x: dfn(A, b, c_vec, x)

    #Initialize error array as err_arr
    err_arr = np.array([])
    counter = 0
    #Compute value to compare against the tolerance for the stopping condition.
    pk = - df(x_0)

    ak = backtrack_line_search(f, df, A, b, c, rho, x_0, pk)
    xk = x_0 + ak * pk
    while linalg.norm(pk) > eps:

        err_arr = np.append(err_arr, f(xk))
        counter += 1

        #Calculate descent direction.
        pk = - df(xk)
        #Determine step size through backtracking line search.
        ak = backtrack_line_search(f, df, A, b, c, rho, xk, pk)
        #Update xk
        xk = xk + ak * pk

    err_arr = err_arr - f(xk)
    return np.array([xk, err_arr, counter])


def newton(fn, dfn, hessfn, A, b, c_vec, c, rho, x0, eps):
    """
    Implementation of Newton's method.
    Follows pseudocode presented in lecture.
    Parameters:
        --fn: function with vector in R^{100}, takes parameters A,b, and c_vec
        - dfn: gradient of function f, takes vector in R^{100} as input
                and paramters A, b, and c_vec
        - hessfn: hessian of function f, takes vector in R^{100} as input
                and parameters A, b, and c_vec
        - A: 500x100 real-valued matrix
        - b: vector in R^{500}
        - c_vec: vector in R^{100}
        - x0: vector in R^{100} taken as input
        - eps: tolerance level
    Returns list featuring:
        -list[0]: xk, which is the approximation of the minimizer
        -list[1]: err_arr, which is the array of errors in the approximation
        -list[2]: counter, which is number of iterations.
    """
    f = lambda x: func(A, b, c_vec, x)
    df = lambda x: dfunc(A, b, c_vec, x)
    hessf = lambda x: hess_func(A, b, c_vec, x)

    #Initialize counter and error array
    err_arr = np.array([])
    counter = 0

    xk = x0
    #Compute descent direction and lambda_k2 (stopping condition)

    dfx = df(xk)
    hessfx = hessf(xk)
    pk = - np.dot(linalg.inv(hessfx), dfx)
    ak = backtrack_line_search(f, df, A, b, c, rho, xk, pk)
    xk = xk + ak * pk
    lambda_k2 = np.dot(np.dot(np.transpose(dfx), hessfx), dfx)

    while (lambda_k2 / 2) > eps:

        err_arr = np.append(err_arr, f(xk))
        counter += 1

        #Newton Step
        #Recalculate descent direction
        dfx = df(xk)
        hessfx = hessf(xk)
        pk = - np.dot(linalg.inv(hessfx), dfx)

        #Recalculate step size through backtracking line search
        ak = backtrack_line_search(f, df, A, b, c, rho, xk, pk)

        #Recalculate lambda_k2
        lambda_k2 = np.dot(np.dot(np.transpose(dfx), hessfx), dfx)
        #Update xk
        xk = xk + ak * pk

    err_arr = err_arr - f(xk)
    return np.array([xk, err_arr, counter])


def gen():
    """
    Generates A, b, c_vec for the specified objective function.
    Also generates randomly starting vector x0 as well as well as c, rho.
    Parameters:
        -none
    Returns all generated parameters in a list:
        -0: A
        -1: b
        -2: c_vec
        -3: x0
        -4: c
        -5: rho
    """
    A = np.random.randn(500, 100)
    c_vec = np.random.randn(100, 1)
    x0 = np.random.randn(100, 1)

    #generate b based on the generated starting vector x0
    b = np.dot(A, x0) + 2 * np.random.rand(500, 1)

    c = np.random.uniform(0, 1)
    rho = np.random.uniform(0, 1)

    return [A, b, c_vec, x0, c, rho]

#################### PART B: ERROR PLOTS ####################
lst = gen()
A = lst[0]
b = lst[1]
c_vec = lst[2]
x0 = lst[3]

#################### C ####################
list_c = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90])
#Constants
rho = 0.05
eps = 0.001
label_sd = np.array(["c = 0.01", "c = 0.05", "c = 0.10", "c = 0.25",
                        "c = 0.50", "c = 0.75", "c = 0.90"])

#Steepest Descent
log_err_sd = [ 0 ] * 7
counter_sd = [ 0 ] * 7
label_sd = np.array(["c = 0.01", "c = 0.05", "c = 0.10", "c = 0.25",
                        "c = 0.50", "c = 0.75", "c = 0.90"])
for i in range(0, 7):
    #array for which each entry is a list of the form [xk, err_arr, counter]
    sd_output = steepest_descent(func, dfunc, A, b, c_vec, list_c[i], rho, x0, eps)
    log_err_sd[i] = np.log(sd_output[1]).transpose()
    count = sd_output[2]
    counter_sd[i] = np.array(range(0, count)).reshape([count, 1])

for i in range(0, 7):
    plt.plot(counter_sd[i], log_err_sd[i], label=label_sd[i])
    plt.title("Graph of Steepest Descent Log Error")
    plt.xlabel("k")
    plt.ylabel("Log Error of the kth iteration")
    plt.legend()
    plt.grid(True)
    plt.show()
'''

'''
#Newton
log_err_newton = [ 0 ] * 7
counter_newton = [ 0 ] * 7

for i in range(0, 7):
    #array for which each entry is a list of the form [xk, err_arr, counter]
    newton_output = newton(func, dfunc, hess_func,
                            A, b, c_vec, list_c[i], rho, x0, eps)
    log_err_newton[i] = np.log(newton_output[1]).transpose()
    count = newton_output[2]
    counter_newton[i] = np.array(range(0, count)).reshape([count, 1])

for i in range(0, 7):
    plt.plot(counter_newton[i], log_err_newton[i], label=label_sd[i])
    plt.title("Graph of Newton's Algorithm Log Error")
    plt.xlabel("k")
    plt.ylabel("Log Error of the kth iteration")
    plt.legend()
    plt.grid(True)
    plt.show()


#################### RHO ####################
list_rho = np.array([0.05, 0.25, 0.50, 0.75, 0.95])
#Constants
c = 0.01
eps = 0.001

#Steepest Descent
log_err_sd = [ 0 ] * 5
counter_sd = [ 0 ] * 5
label_sd = np.array(["rho = 0.05", "rho = 0.25",
                        "rho = 0.50", "rho = 0.75", "rho = 0.95"])

for i in range(0, 5):
    #array for which each entry is a list of the form [xk, err_arr, counter]
    sd_output = steepest_descent(func, dfunc,
                                    A, b, c_vec, c, list_rho[i], x0, eps)
    log_err_sd[i] = np.log(sd_output[1]).transpose()
    count = sd_output[2]
    counter_sd[i] = np.array(range(0, count)).reshape([count, 1])

for i in range(0, 5):
    plt.plot(counter_sd[i], log_err_sd[i], label=label_sd[i])
    plt.title("Graph of Steepest Descent Log Error")
    plt.xlabel("k")
    plt.ylabel("Log Error of the kth iteration")
    plt.legend()
    plt.grid(True)
    plt.show()

#Newton
log_err_newton = [ 0 ] * 5
counter_newton = [ 0 ] * 5

for i in range(0, 5):
    #array for which each entry is a list of the form [xk, err_arr, counter]
    newton_output = newton(func, dfunc, hess_func,
                                    A, b, c_vec, c, list_rho[i], x0, eps)
    log_err_newton[i] = np.log(newton_output[1]).transpose()
    count = newton_output[2]
    counter_newton[i] = np.array(range(0, count)).reshape([count, 1])

for i in range(0, 5):
    plt.plot(counter_newton[i], log_err_newton[i], label=label_sd[i])
    plt.title("Graph of Newton's Algorithm Log Error")
    plt.xlabel("k")
    plt.ylabel("Log Error of the kth iteration")
    plt.legend()
    plt.grid(True)
    plt.show()



#################### EPSILON ####################
list_eps = np.array([0.001, 0.00001, 0.00000001])
#Constants
c = 0.01
rho = 0.05

#Steepest Descent
sd_output = steepest_descent(func, dfunc, A, b, c_vec, c, rho, x0, 0.00001)
log_err_sd = np.log(sd_output[1]).transpose()
count = sd_output[2]
counter_sd = np.array(range(0, count)).reshape([count, 1])

plt.plot(counter_sd, log_err_sd, label="eps = 0.00001")
plt.title("Graph of Steepest Descent Log Error")
plt.xlabel("k")
plt.ylabel("Log Error of the kth iteration")
plt.legend()
plt.grid(True)
plt.show()


#Newton
newton_output = newton(func, dfunc, hess_func,
                                A, b, c_vec, c, rho, x0, 0.00001)
log_err_newton = np.log(newton_output[1]).transpose()
count = newton_output[2]
counter_newton = np.array(range(0, count)).reshape([count, 1])

plt.plot(counter_newton, log_err_newton, label="eps = 0.00001")
plt.title("Graph of Newton's Algorithm' Log Error")
plt.xlabel("k")
plt.ylabel("Log Error of the kth iteration")
plt.legend()
plt.grid(True)
plt.show()

#Newton
newton_output = newton(func, dfunc, hess_func,
                                A, b, c_vec, c, rho, x0, 0.00000001)
log_err_newton = np.log(newton_output[1]).transpose()
count = newton_output[2]
counter_newton = np.array(range(0, count)).reshape([count, 1])

plt.plot(counter_newton, log_err_newton, label="eps = 0.00000001")
plt.title("Graph of Newton's Algorithm' Log Error")
plt.xlabel("k")
plt.ylabel("Log Error of the kth iteration")
plt.legend()
plt.grid(True)
plt.show()
















