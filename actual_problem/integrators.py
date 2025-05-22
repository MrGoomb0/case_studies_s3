"""
This file contains all the integrators that can be used to compute the 2D problem.

To create a new integrator, just follow the same skeleton as the function described below, 
as this will ensure that it works well with the other functions.
"""

from casadi import *

"""
Returns a integrator based on the implicit Crank Nicholson method, using the built-in 'rootfinder' function.
---
## Parameters:
 - f : derivative function
 - T : end time
 - N : number of time steps
 - M : amount of iterations per time step

 ## Return
 - F : integrator that can be used to solve the 2D problem
"""
def cranknicholson(f, T, N, M=1):
    h = T / N / M
    
    X0 = MX.sym('X0', 5)
    J0 = MX.sym('J0')
    Y0 = vertcat(X0, J0)
    T0 = MX.sym('T0')
    Y = MX.sym('Y', 6)
    U = MX.sym('U')
    
    Yk = Y0
    Tk = T0
    I = Function('I', [Y, U, Tk, Yk], [Y - Yk - h / 2 * (f(Y, U, Tk) + f(Yk, U, Tk + h))], ['y0', 'p', 'tk', 'yk'], ['If'])

    G = rootfinder('G', 'fast_newton', I)
    
    Y_guess = Yk
    for j in range(M):
        Y_guess = G(Y_guess, U, Tk, Yk)
    Yk = Y_guess
    Tk = Tk + h
    Xk = Y_guess[0:5]
    Jk = Y_guess[5]

    F = Function('F', [X0, J0, T0, U], [Xk, Jk, Tk], ['x0', 'j0', 't0', 'p'], ['xf', 'jf', 'tf'])
    return F

"""
Returns a integrator based on the explicit Euler method.
---
## Parameters:
 - f : derivative function
 - T : end time
 - N : number of time steps
 - M : amount of iterations per time step

 ## Return
 - F : integrator that can be used to solve the 2D problem
"""
def expEuler(f, T, N, M):
    h = T / N / M

    X0 = MX.sym('X0', 5)
    J0 = MX.sym('J0')
    Y0 = vertcat(X0, J0)
    T0 = MX.sym('T0')
    U = MX.sym('U')
    
    Yk = Y0
    Tk = T0
    Jk = J0

    for i in range(M):
        k1 = f(Yk, U, Tk)
        Yk = Yk + h*k1 
        Tk = Tk + h
    Xk = Yk[0:5]
    Jk = Yk[5]

    F = Function('F', [X0, J0, T0, U], [Xk, Jk, Tk], ['x0', 'j0', 't0', 'p'], ['xf', 'jf', 'tf'])
    return F

"""
Returns a integrator based on the classic 4 stage runga kutta method.
---
## Parameters:
 - f : derivative function
 - T : end time
 - N : number of time steps
 - M : amount of iterations per time step

 ## Return
 - F : integrator that can be used to solve the 2D problem
"""
def rungakutta(f, T, N, M):
    h = T / N / M

    X0 = MX.sym('X0', 5)
    J0 = MX.sym('J0')
    Y0 = vertcat(X0, J0)
    T0 = MX.sym('T0')
    U = MX.sym('U')

    Yk = Y0
    Tk = T0
    Jk = J0

    for j in range(M):
        k1 = f(Yk, U, Tk)
        k2 = f(Yk + h/2 * k1, U, Tk + h/2)
        k3 = f(Yk + h/2 * k2, U, Tk + h/2)
        k4 = f(Yk + h * k3, U, Tk  + h)
        Yk = Yk + h/6*(k1 +2*k2 +2*k3 +k4)
    Xk = Yk[0:5]
    Jk = Yk[5]

    F = Function('F', [X0, J0, T0, U], [Xk, Jk, Tk], ['x0', 'j0', 't0', 'p'], ['xf', 'jf', 'tf'])
    return F

"""
Returns a integrator based on the Dormanad-Prince 6 stage runga kutta method of order 5.
---
## Parameters:
 - f : derivative function
 - T : end time
 - N : number of time steps
 - M : amount of iterations per time step

 ## Return
 - F : integrator that can be used to solve the 2D problem
"""
def rungakutta6(f, T, N, M):
    h = T / N / M

    # X0 = MX.sym('X0', 5)
    X0 = MX.sym('X0')
    J0 = MX.sym('J0')
    Y0 = vertcat(X0, J0)
    T0 = MX.sym('T0')
    U = MX.sym('U')

    Yk = Y0
    Tk = T0
    Jk = J0

    for j in range(M):
        k1 = f(Yk, U, Tk)
        k2 = f(Yk + h * (1/5 * k1), U, Tk + h * 1/5)
        k3 = f(Yk + h * (3/40 * k1 + 9 / 40 * k2), U, Tk + h * 3/10)
        k4 = f(Yk + h * (44/55 * k1 - 56/15 * k2 + 32/9 * k3), U, Tk  + h * 4/5)
        k5 = f(Yk + h * (19372 / 6561 * k1 - 25360 / 2187 * k2 + 64448 / 6561 * k3 - 212/729 * k4), U, Tk + h * 8/9)
        k6 = f(Yk + h * (9017/3168 * k1 - 355/33 * k2 + 46732/5247 * k3 + 49/176 * k4 - 5103/18656 * k5), U, Tk + h)
        Yk = Yk + h * (35/384 * k1 + 500/1113 * k3 + 125/192 * k4 - 2187/6784 * k5 + 11/84 * k6)
    Xk = Yk[0:5]
    Jk = Yk[5]

    F = Function('F', [X0, J0, T0, U], [Xk, Jk, Tk], ['x0', 'j0', 't0', 'p'], ['xf', 'jf', 'tf'])
    return F