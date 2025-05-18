"""
Implementation of the Example 4.6.4 in 'Optimal Control of ODEs and DAEs' by Matthis Gerdts.
"""
from casadi import *
import numpy as np
import matplotlib.pyplot as plt

def main():
    T_f = 1
    N = 100

    x = MX.sym('x')

    u = MX.sym('u')
    t = MX.sym('t')

    xdot = u - 15*exp(-2*t)
    L_1 = 0.5 * u**2 + 0.5 * x**3

    f = Function('f', [x, u, t], [xdot, L_1])

    F = rungakutta(f, T_f, N)
    u_opt = nlpSolver(N, F)
    plotSolution(u_opt, T_f, N, F)


"""
Function that returns an integrator given the derivative function
---
 - f: The derivative function
 - T_f: End time
 - N: Number of steps
"""
def rungakutta(f, T_f, N):
    h = T_f/N
    X0 = MX.sym('X0')
    T0 = MX.sym('T0')
    U = MX.sym('U')
    X = X0
    T = T0
    Q = 0

    k1, k1_E = f(X, U, T)
    k2, k2_E = f(X + h/2 * k1, U, T + h/2)
    k3, k3_E = f(X + h/2 * k2, U, T + h/2)
    k4, k4_E = f(X + h * k3, U, T + h)
    X = X + h/6*(k1 +2*k2 +2*k3 +k4)
    Q = Q + h/6*(k1_E +2*k2_E +2*k3_E +k4_E)
    T = T + h

    F = Function('F', [T0, X0, U], [T, X, Q], ['t0', 'x0', 'p'], ['tf', 'xf', 'pf'])
    return F

"""
Solves the problem for a given number of steps and an integrater.
---
 - N: Number of steps
 - F: Integrator
"""
def nlpSolver(N, F):
    w=[]
    w0 = []
    lbw = []
    ubw = []
    J = 0

    Tk = MX(0)
    Xk = MX(4)

    for k in range(N):
        Uk = MX.sym('U_' + str(k))
        w += [Uk]
        lbw += [-inf]
        ubw += [inf]
        w0 =  [0]

        Fk = F(t0=Tk, x0=Xk, p=Uk)
        Xk = Fk['xf']
        Tk = Fk['tf']
        J = J + Fk['pf']

    J = J + 5/2 * (Xk - 1)**2

    prob = {'f': J, 'x': vertcat(*w)}
    solver = nlpsol('solver', 'ipopt', prob)

    sol = solver(x0=w0, lbx=lbw, ubx=ubw)
    w_opt = sol['x']
    return w_opt

"""
Plots the solution to the problem.
---
 - u_opt: Calculated solution
 - T: End time
 - N: Number of steps
 - F: Integrator
"""
def plotSolution(u_opt, T_f, N, F):
    x_opt = [4]
    t_opt = [0]
    for k in range(N):
        Fk = F(t0=t_opt[-1], x0=x_opt [-1], p=u_opt[k])
        x_opt += [Fk['xf']]
        t_opt += [Fk['tf']]
    tgrid = [T_f/N*k for k in range(N + 1)]
    x_opt = vcat(x_opt)
    plt.plot(tgrid, np.array(x_opt), '--')
    plt.step(tgrid, vertcat(DM.nan(1), u_opt), '-,')
    plt.xlabel('t')
    plt.legend(['x','u'])
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()