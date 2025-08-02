"""
Implementation of the Example 1.0.1 in 'Optimal Control of ODEs and DAEs' by Matthis Gerdts.
"""
from casadi import *
import numpy as np
import matplotlib.pyplot as plt 

ALPHA = 45 / 180 * pi
LENGTH = 5
U_MAX = 1

T = 1

N = 200
M = 1

def main():
    x1 = MX.sym('x1')
    x2 = MX.sym('x2')
    x3 = MX.sym('x3') #Curve length
    u = MX.sym('u')
    x = vertcat(x1, x2, x3)
    xdot1 = x2
    xdot2 = u
    xdot3 = sqrt(1 + x2**2)
    xdot = vertcat(xdot1, xdot2, xdot3)
    L = MX.sym('L')
    Ldot = u**2

    f = Function('f', [x, u], [xdot, Ldot])

    F = rungaKutta(f=f, N=N, M=M, T=T)

    x0 = np.array([0, tan(ALPHA), LENGTH]).reshape(3)
    U = np.ones(N) * 5
    init_estimate = initStateUsingExpEuler(f=f, N=N, x0=x0, u=U)

    u_opt, x_opt = nlpsolver(N=N, F=F, init_estimate=init_estimate, u_init_estimate=U)
    grid = np.linspace(0, 1, N + 1)

    X_TARGET = 0.03
    STEP = 0.01
    X_MAX = np.arange(0.1, X_TARGET, -STEP)
    solutions = np.zeros((len(X_MAX), N + 1, 3))
    for i, x_max in enumerate(X_MAX):
        u_opt, x_opt = nlpsolver(N=N, F=F, init_estimate=x_opt, u_init_estimate=u_opt, x_min=0, x_max=x_max, u_max=U_MAX)
        solutions[i, :, :] = x_opt  
        plt.plot(grid, x_opt[:, 0], '-')
    plt.plot(grid, x_opt[:, 0], '-')
    legend = [str(i) for i in X_MAX]
    print(legend)
    plt.legend()
    plt.show()

"""
Solves the problem for a given number of steps and an integrater.
---
 - N: Number of steps
 - F: Integrator
 - init_estimate: initial estimate for the state variables.
 - u_init_estimate: initial estimate for the control variables
 - x_min: minimum value for the x1 state variable
 - x_max: maximum value for the x1 state variable
 - u_max: maximum value for the control variable
"""
def nlpsolver(N, F, init_estimate, u_init_estimate, x_min=0, x_max=inf, u_max=inf):
    Jk = 0
    w = []
    lbw = []
    ubw = []
    w0 = []

    g = []
    lbg = []
    ubg = []

    Xk =  MX.sym('X0', 3)
    w += [Xk]
    lbw += init_estimate[0].tolist()
    ubw += init_estimate[0].tolist()
    w0 += init_estimate[0].tolist()
    

    for k in range(N):
        Uk = MX.sym('U_' + str(k))
        w += [Uk]
        lbw += [-u_max]
        ubw += [u_max]
        w0 += [u_init_estimate[k]]

        Fk = F(x0=Xk, p=Uk, l0=Jk)
        Jk = Fk['lf']
        Xk_end = Fk['xf']

        Xk = MX.sym('X_' + str(k + 1), 3)
        w += [Xk]
        if k == N-1:
            lbw += [x_min, -tan(ALPHA), LENGTH]
            ubw += [x_max, -tan(ALPHA), LENGTH]
        else:
            lbw += [x_min, -inf, LENGTH]
            ubw += [x_max, inf, LENGTH]
        w0 += init_estimate[k + 1, :].tolist()

        g += [Xk_end - Xk]
        lbg += [0, 0, 0]
        ubg += [0, 0, 0]

    J = Jk
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', prob)
        
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x'].full().flatten()
    u_opt = np.array([w_opt[i] for i in range(2, 4*N + 3, 4)])
    x_opt = np.array([[w_opt[i], w_opt[i + 1], w_opt[i + 2]]  for i in range(0, 4*N + 3, 4)])
    return u_opt, x_opt  

"""
Calculates the an initial guess for the state variables given the derivative funciton 'f' and the series of control variable 'u'.
---
 - f : derivative function
 - N : number of time steps
 - x0 : initial values for the state variables for t=0
 - u : initial values for the control variable for all time steps
"""
def initStateUsingExpEuler(f, N, x0, u):
    h = T / N
    
    Xk = x0
    solution = np.zeros((N + 1, 3))
    solution[0, :] = Xk
    for i in range(N):
        k1, k1_L = f(Xk, u[i])
        Xk = Xk + h * k1
        solution[i + 1, :]  = np.array(Xk).reshape(3)

    return solution


"""
Function that returns a classic 4step Runga Kutta integrator given the derivative function
---
 - f: The derivative function
 - T: End time
 - N: Number of time steps
 - M: Number of substeps per time step
"""
def rungaKutta(f, N, M, T):
    h =  T / N / M

    X0 = MX.sym('X0', 3)
    U0 = MX.sym('U0')

    L0 = MX.sym('L0')

    Xk = X0
    Uk = U0
    Lk = L0


    for i in range(M):
        k1, k1_L = f(Xk, Uk)
        k2, k2_L = f(Xk + h/2 * k1, Uk)
        k3, k3_L = f(Xk + h/2 * k1, Uk)
        k4, k4_L = f(Xk + h * k3, Uk)
        Xk = Xk + h * (1/6 * k1 + 2/6 * k2 + 2/6 * k3 + 1/6 * k4)
        Lk = Lk + h * (1/6 * k1_L + 2/6 * k2_L + 2/6 * k3_L + 1/6 * k4_L)

    F = Function('F', [X0, U0, L0], [Xk, Lk], ['x0', 'p', 'l0'], ['xf', 'lf'])
    return F


if __name__ == "__main__":
    main()