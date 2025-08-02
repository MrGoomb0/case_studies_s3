"""
Implementation of the Example 5.1.6 in 'Optimal Control of ODEs and DAEs' by Matthis Gerdts.
"""
from casadi import *
import numpy as np
import matplotlib.pyplot as plt

EXPERIMENT_1 = False
EXPERIMENT_2 = False
EXPERIMENT_3 = False

def main():
    #Problem set-up
    T = 1
    x = MX.sym('x')
    u = MX.sym('u')

    xdot = 0.5*x +  u
    L = u**2 + 2*x**2

    f = Function('f_x', [x, u], [xdot, L])

    # Non-convergence for Euler integrator if 'u_n' and 'u_n1/2' is used.
    if EXPERIMENT_1:
        N = 40
        F = rungakutta(f, T=T, N=N, method='euler', step_wise=False)
        u_opt = nlpSolver(N=N, F=F, step_wise=False)
        plotSolution(u_opt=u_opt, T=T, N=N, F=F, step_wise=False, plot_x=False, plot_u_e=False, plot_x_e=False)
    
    # Error and order calculation for Euler integrator
    if EXPERIMENT_2:
        N_list = [10, 20, 40, 80, 160, 320]
        errors = np.zeros(len(N_list))
        step_wise=True
        for i, N in enumerate(N_list):
            F = rungakutta(f, T=T, N=N, method='euler', step_wise=step_wise)
            u_opt = nlpSolver(N=N, F=F, step_wise=step_wise)
            errors[i] = calculateError(u_opt, N, F, step_wise=step_wise)
        orders = np.zeros(len(N_list) - 1)
        for i in range(len(N_list) - 1):
            orders[i] = errors[i] / errors[i + 1] * N_list[i] / N_list[i + 1]
        print("\n")
        for i in range(len(N_list)):
            if i == 0:
                print("N: ", N_list[i], "Error: ", errors[i], "Order: -")
            else:
                print("N: ", N_list[i], "Error: ", errors[i], "Order: ", orders[i-1])

    # Error and order calculation for Heun integrator
    if EXPERIMENT_3:
        N_list = [10, 20, 40, 80, 160, 320]
        errors = np.zeros(len(N_list))
        step_wise=True
        for i, N in enumerate(N_list):
            F = rungakutta(f, T=T, N=N, method='heun', step_wise=step_wise)
            u_opt = nlpSolver(N=N, F=F, step_wise=step_wise)
            errors[i] = calculateError(u_opt, N, F, step_wise=step_wise)
        orders = np.zeros(len(N_list) - 1)
        for i in range(len(N_list) - 1):
            orders[i] = errors[i] / errors[i + 1] * N_list[i] / N_list[i + 1]
        print("\n")
        for i in range(len(N_list)):
            if i == 0:
                print("N: ", N_list[i], "Error: ", errors[i], "Order: -")
            else:
                print("N: ", N_list[i], "Error: ", errors[i], "Order: ", orders[i-1])

"""
Function that returns an integrator given the derivative function
---
 - f: The derivative function
 - T: End time
 - N: Number of steps
 - method: Type of integrator
    ex: 'euler', 'heun'
 - step_wise: Indicates if only 'u_n' is used, or also 'u_n1/2'
"""
def rungakutta(f, T, N, method="euler", step_wise=True):
    h = T / N
    
    X0 = MX.sym('X0')
    if step_wise:
        U = MX.sym('U')
    else:
        U1 = MX.sym('u1')
        U2 = MX.sym('U2')
    
    X = X0
    E = 0

    match method:
        case 'euler':
            if step_wise:
                k1, k1_E = f(X, U)
                k2, k2_E = f(X + h/2*k1, U)
                X = X + h*k2
                E = E + h*k2_E
                F = Function('F', [X0, U], [X, E], ['x0', 'p'], ['xf', 'pf'])
            else:
                k1, k1_E = f(X, U1)
                k2, k2_E = f(X + h/2*k1, U2)
                X = X + h*k2
                E = E + h*k2_E
                F = Function('F', [X0, U1, U2], [X, E], ['x0', 'p1', 'p2'], ['xf', 'pf'])
        case 'heun':
            if step_wise:
                k1, k1_E = f(X, U)
                k2, k2_E = f(X + h*k1, U)
                X = X + h/2*(k1 + k2)
                E = E + h/2*(k1_E + k2_E)
                F = Function('F', [X0, U], [X, E], ['x0', 'p'], ['xf', 'pf'])
            else:
                k1, k1_E = f(X, U1)
                k2, k2_E = f(X + h*k1, U2)
                X = X + h/2*(k1 + k2)
                E = E + h/2*(k1_E + k2_E)
                F = Function('F', [X0, U1, U2], [X, E], ['x0', 'p1', 'p2'], ['xf', 'pf'])
        case '_':
            raise Exception("WARNING: " + str(method) + " is an uninitialised method for integrator.")
    return F

"""
Solves the problem for a given number of steps and an integrater.
---
 - N: Number of steps
 - F: Integrator
 - step_wise: Indicates if only 'u_n' is used, or also 'u_n1/2'
"""
def nlpSolver(N, F, step_wise=True):
    w=[]
    if step_wise:
        w0 = np.zeros(N)
    else:
        w0 = np.zeros(2*N)
    J = 0
    Xk = MX(1)
    for k in range(N):
        if step_wise:
            Uk = MX.sym('U_' + str(k))
            w += [Uk]
            Fk = F(x0=Xk, p=Uk)
        else:
            Uk = MX.sym('U_' + str(k))
            w += [Uk]
            Uk_2 = MX.sym('U_' + str(k + 0.5))
            w += [Uk_2]
            Fk = F(x0=Xk, p1=Uk, p2=Uk_2)
        Xk = Fk['xf']
        J = J + Fk['pf']
    
    prob = {'f': J, 'x': vertcat(*w)}
    solver = nlpsol('solver', 'ipopt', prob)

    sol = solver(x0=w0)
    w_opt = sol['x']
    return w_opt

"""
Plots the solution to the problem.
---
 - u_opt: Calculated solution
 - T: End time
 - N: Number of steps
 - F: Integrator
 - step_wise: Indicates if only 'u_n' is used, or also 'u_n1/2'
 - plot_u: Plot 'u'
 - plot_x: Plot 'x'
 - plot_u_e: Plot the exact 'u'
 - plot_x_e: Plot the exact 'x'
"""
def plotSolution(u_opt, T, N, F, step_wise, plot_u=True, plot_x=True, plot_u_e=True, plot_x_e=True):
    x_opt = [1]
    for k in range(N):
        if step_wise:
            Fk = F(x0=x_opt[-1], p=u_opt[k])
        else:
            Fk = F(x0=x_opt[-1], p1=u_opt[2*k], p2=u_opt[2*k + 1])
        x_opt += [Fk['xf']]
    x_opt = vcat(x_opt)

    tgrid_x = [T/N*k for k in range(N +1)]
    if step_wise:
        tgrid_u = tgrid_x
    else:
        tgrid_u = [T/N*k/2 for k in range(2*N)]
    
    legend_entries = []
    if plot_x:
        plt.plot(tgrid_x, x_opt, '--')
        legend_entries.append('x')
    if plot_u:
        if step_wise:
            plt.step(tgrid_u, vertcat(DM.nan(1), u_opt), '-.')
        else:
            plt.plot(tgrid_u, u_opt)
        legend_entries.append('u')
    if plot_u_e:
        t = np.linspace(0, 1, 100)
        u_e = (2*(np.exp(3*t) - np.exp(3))) / (np.exp(3*t/2) * (2 + exp(3)))
        plt.plot(t, u_e, "-")
        legend_entries.append('u_e')
    if plot_x_e:
        t = np.linspace(0, 1, 100)
        x_e = ((2 * np.exp(3*t)) + np.exp(3)) / (np.exp(3*t/2) * (2 + exp(3)))
        plt.plot(t, x_e, "-")
        legend_entries.append('x_e')

    plt.xlabel('t')
    plt.legend(legend_entries)
    plt.grid()
    plt.show()

def calculateError(u_opt, N, F, step_wise):
    x_opt = [1]
    for k in range(N):
        if step_wise:
            Fk = F(x0=x_opt[-1], p=u_opt[k])
        else:
            Fk = F(x0=x_opt[-1], p1=u_opt[2*k], p2=u_opt[2*k + 1])
        x_opt += [Fk['xf']]
    x_opt = vcat(x_opt)
    t = np.linspace(0, 1, N+1)
    x_e = ((2 * np.exp(3*t)) + np.exp(3)) / (np.exp(3*t/2) * (2 + exp(3)))
    error = np.max(abs(np.reshape(np.array(x_opt), N+1) - x_e))
    return error


if __name__ == "__main__":
    main()


