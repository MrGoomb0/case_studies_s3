from casadi import *
import numpy as np 
import matplotlib.pyplot as plt

def main():
    T = 1
    x1 = MX.sym('x1') # h
    x2 = MX.sym('x2') # v
    x3 = MX.sym('x3') # m
    x = vertcat(x1 ,x2, x3)

    u = MX.sym('u')

    D = 310 * x2**2 * exp(500 * (1 - x1))

    x1dot = x2
    x2dot = 1 / x3 * (u - D) - 1 / x1**2
    x3dot = -2 * u
    xdot = vertcat(x1dot, x2dot, x3dot)

    f = Function('f', [x, u], [xdot])

    N = 200

    F = cranknicholson(f=f, T=T, N=N)


    u_opt, x1_opt, x2_opt, x3_opt = nlpSolver(N=N, F=F)

    u_t = np.linspace(0, 1, N)
    t = np.linspace(0, 1, N+1)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(u_t, u_opt)
    ax2.plot(t, x1_opt)
    ax3.plot(t, x2_opt)
    ax4.plot(t, x3_opt)
    plt.show()

"""
Function that returns a classic 4step Runga Kutta integrator given the derivative function
---
 - f: The derivative function
 - T: End time
 - N: Number of steps
"""
def rungakutta(f, T, N):
    M = 1
    h = T/N/M
    X0 = MX.sym('X0', 3)
    U = MX.sym('U')
    X = X0
    for j in range(M):
        k1 = f(X, U)
        k2 = f(X + h/2 * k1, U)
        k3 = f(X + h/2 * k2, U)
        k4 = f(X + h * k3, U)
        X=X+h/6*(k1 +2*k2 +2*k3 +k4)
    
    F = Function('F', [X0, U], [X], ['x0', 'p'], ['xf'])
    return F

"""
Function that returns an Crank-Nicholson integrator given the derivative function
---
 - f: The derivative function
 - T: End time
 - N: Number of steps
"""
def cranknicholson(f, T, N):
    M = 5
    h = T / N / M
    X0 = MX.sym('X0', 3)
    X = MX.sym('X', 3)
    U = MX.sym('U')
    Xk = X0
    I = Function('I', [X, U, Xk], [X - Xk - h / 2 * (f(X, U) + f(Xk, U))], ['x0', 'p', 'xk'], ['If'])
    G = rootfinder('G', 'newton', I)
    X_guess = Xk
    for j in range(M):
        X_guess = G(X_guess, U, Xk)
    Xk = X_guess

    F = Function('F', [X0, U], [Xk], ['x0', 'p'], ['xf'])
    return F

"""
Solves the problem for a given number of steps and an integrater.
---
 - N: Number of steps
 - F: Integrator
 - step_wise: Indicates if only 'u_n' is used, or also 'u_n1/2'
"""
def nlpSolver(N, F):
    w = []
    w0 = []
    lbw, ubw = [], []
    J = 0
    g = []
    lbg, ubg = [], []

    Xk = MX.sym('X0', 3)
    w += [Xk]
    lbw += [1, 0, 1]
    ubw += [1, 0, 1]
    w0 += [1, 0, 1]

    for k in range(N):
        Uk = MX.sym('U_' + str(k))
        w   += [Uk]
        lbw += [0]
        ubw += [3.5]
        w0  += [3.5]

        Fk = F(x0=Xk, p=Uk)
        Xk_end = Fk['xf']
        
        Xk = MX.sym('X_' + str(k+1), 3)
        w   += [Xk]
        if k == N-1:
            lbw += [-inf, -inf, 0.6]
            ubw += [inf, inf, 0.6]
        else:
            lbw += [-inf, -inf, -inf]
            ubw += [inf, inf, inf]
        w0  += [1, 0, 1]

        g   += [Xk_end-Xk]
        lbg += [0, 0, 0]
        ubg += [0, 0, 0]

    J = -Xk[0]
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', prob)

    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x'].full().flatten()
    u_opt = [w_opt[i] for i in range(3, 4*N+3, 4)]
    x1_opt = [w_opt[i] for i in range(0, 4*N+3, 4)]
    x2_opt = [w_opt[i] for i in range(1, 4*N+3, 4)]
    x3_opt = [w_opt[i] for i in range(2, 4*N+3, 4)]
    return u_opt, x1_opt, x2_opt, x3_opt
    return w_opt
if __name__ == "__main__":
    main()