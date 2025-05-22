"""
This file contains all the solvers that can be used to compute the 2D problem.

To create a new sovler, just follow the same skeleton as the function described below, 
as this will ensure that it works well with the other functions.
"""

from casadi import *
from .integrators import expEuler

# Constants used in the problem defined by Pesch et al.

T_F = 40

ALPHA_MAX = 17.2 / 180 * pi      # Has to be in rad.

GAMMA_F = 7.431 / 180 * pi       # Has to be in rad.

U_MAX = 3 / 180 * pi             # Has to be in rad.
U_MIN = - U_MAX

"""
Initial implementation of a simple solver that uses multiple shooting to solve the 2D problem.
---
## Parameters
 - N : number of time steps
 - F : integrator
 - initial_estimate : initial estimate for the problem np.array with size = (N + 1, 5) (use for homopoty)
 - gamma_f : the final value of gamma_f (use for homopty)
 - alpha_max : the max value for alpha (use for homopoty)
"""
def nlpsolver(N: int, F: Function, initial_estimate: np.array, u_init_estimate: np.array, t_f=T_F, gamma_f : float =GAMMA_F, alpha_max : float =ALPHA_MAX):
    Tk = 0
    w = []
    w0 = []
    lbw, ubw = [], []
    Jk = 0
    g = []
    lbg, ubg = [], []

    Xk = MX.sym('X0', 5)

    w += [Xk]
    lbw += initial_estimate[0].tolist()
    ubw += initial_estimate[0].tolist()
    w0 += initial_estimate[0].tolist()

    for k in range(N):
        Uk = MX.sym('U_' + str(k))
        w += [Uk]
        lbw += [U_MIN]
        ubw += [U_MAX]
        w0 += [u_init_estimate[k]]

        Fk = F(x0=Xk, j0=Jk, t0=Tk, p=Uk)
        Tk = Fk['tf']
        Xk_end = Fk['xf']
        Jk = Fk['jf']

        Xk = MX.sym('X_' + str(k+1), 5)
        w += [Xk]
        if k == N - 1:
            if gamma_f == inf: # Branch to ignore the gamma_f condition
                lbw += [-inf, -inf, -inf, -inf, -inf]
                ubw += [inf, inf, inf, inf, alpha_max]
            else:
                lbw += [-inf, -inf, -inf, gamma_f, -inf]
                ubw += [inf, inf, inf, gamma_f, alpha_max]
        else:
            lbw += [-inf, -inf, -inf, -inf, -inf]
            ubw += [inf, inf, inf, inf, alpha_max]
        w0 += initial_estimate[k + 1].tolist()

        g += [Xk_end - Xk]
        lbg += [0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0]

    J = Jk
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', prob)
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x'].full().flatten()
    return w_opt


"""
Calculates an initial estimate according to the derivative function 'f' and the given control state values 'u'.
---
## Params
 - f : derivative function
 - N : number of time steps
 - M : amount of iterations per time step
 - x_0 : initial horizontal position state
 - h_0 : initial vertical position state
 - v_0 : initial velocity state
 - gamma_0 : initial gamma state
 - alpha_0 : initial alpha state
 - t_f : end time

 ## Return
  - initial_estimate : np.array with size (N + 1, 5) containing the initial estimate.

"""
def initialEstimatorUsingExpEuler(f: Function, N: int, M: int, u: np.array, x_0: float, h_0: float, v_0: float, gamma_0: float, alpha_0: float, t_f: float=T_F):
    X0 = np.array([x_0, h_0, v_0, gamma_0, alpha_0])
    initial_estimate = np.zeros((N + 1, 5))
    initial_estimate[0, :] = X0

    Tk = 0    
    Xk = X0

    F = expEuler(f, t_f, N, M)
    for i in range(N):
        Fk = F(x0=Xk, j0=0, t0=Tk, p=u[i])
        Xk = Fk['xf']
        Tk = Fk['tf']
        initial_estimate[i + 1, :] = np.array(Xk).reshape((5))
    
    return initial_estimate

