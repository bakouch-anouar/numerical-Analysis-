# let's implemen the newton method to resolve non linear systems ( n dimension )

import numpy as np
import matplotlib.pyplot as plt

# define the function
def f(x):
    # f1 = e^x1 - x2
    # f2 = x1^2 + x2^2 - 16
    return np.array([np.exp(x[0]) - x[1], x[0]**2 + x[1]**2 - 16]) # return a vector

# define the jacobian matrix
def j(x):
    # j1 = e^x1, -1
    # j2 = 2x1, 2x2
    return np.array([[np.exp(x[0]), -1], [2*x[0], 2*x[1]]]) # type: ignore # return a matrix

# define the newton method
def newton(x0, tol, N, f, J):
    x = x0
    # jacobian matrix a initial point
    J = j(x)
    # function at initial point
    F = f(x)
    # h = J^-1 * f
    h = np.linalg.solve(J, F) # solve the linear system
    #display h before the iteration
    print("The solution of the 0 iteration: ", h)
    x = x - h # update the solution
    # iterate
    i = 1
    for i in range(N):
        # jacobian matrix
        J = j(x)
        # function
        F = f(x)
        #display the function of each iteration
        print("The function of iteration ", i, " is: ", F)
        print("The jacobian matrix of iteration ", i, " is: ", J)

        # h = J^-1 * f
        # inverse J then multiply by F
        #display the inverse of jacobian matrix of each iteration
        print("The inverse of jacobian matrix of iteration ", i, " is: ", np.linalg.inv(J))
        h = np.linalg.solve(J, F) 
        x = x - h 
        # display the solution of each iteration
        print("The solution of iteration ", i, " is: ", x)
        # check the tolerance
        if np.linalg.norm(h) < tol:
            return x, i
    return x, i


# define the initial point
x0 = np.array([2.8, 2.8])
# define the tolerance
tol = 1e-10
# define the maximum number of iterations
N = 5

x = [2.8, 2.8]

F = f(x)
k = j(x)

print("The function is: ", F)
print("The jacobian matrix is: ", k)
print("The inverse of jacobian matrix of iteration ", 0, " is: ", np.linalg.inv(k))

# call the newton method
x, i = newton(x0, tol, N, f, k)
print("The solution is: ", x)
print("The number of iterations is: ", i)

# plot the function
x = np.linspace(-3, 7, 100)
y = np.linspace(-3,7, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(X) - Y
Z1 = X**2 + Y**2 - 16;
plt.contour(X, Y, Z, 0, colors='r', linestyles='dashed', linewidths=2)
plt.contour(X, Y, Z1, 0, colors='b', linestyles='dashed', linewidths=2)
# show the solution 
plt.plot(x, y, 'ro')
plt.show()





