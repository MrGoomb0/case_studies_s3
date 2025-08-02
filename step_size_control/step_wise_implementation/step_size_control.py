
import numpy as np
from casadi import sqrt, sum, evalf

"""
Given two solutions, one with 'N' points and the other with '2*N - 1' points, 
calculates where the subdivision is necessary for convergence,
and returns the new time discretisation.
"""
def refineSolution(sol1, sol2, ts1, ts2, tol, p):
    if isinstance(sol1, dict):
        sol1 = [sol1]
        sol2 = [sol2]
    N = len(ts1)
    assert len(ts2) == 2*N -1 
    ts_new = [ts1[0]]
    for i in range(1, N):
        for j in range(len(sol1)):
            xk1 = np.array([sol1[j]['x'][i], sol1[j]['h'][i], sol1[j]['V'][i], sol1[j]['gamma'][i], sol1[j]['alpha'][i]])
            xk2 = np.array([sol2[j]['x'][2*i], sol2[j]['h'][2*i], sol2[j]['V'][2*i], sol2[j]['gamma'][2*i], sol2[j]['alpha'][2*i]])
            s = np.linalg.norm(xk1 - xk2) / (2**p - 1)
            dt = ts1[i] - ts1[i - 1]
            q = s / (dt * tol)
            if q > 1:
                ts_new.append(ts2[2*i - 1])
                break
        ts_new.append(ts1[i])
    return np.array(ts_new)

"""
Returns a new discretisation, where each subinterval is divided by 2.
"""
def reduceIntervalBy2(ts):
    ts_new = np.zeros(ts.shape[-1] * 2 - 1)
    for i in range(ts.shape[-1] - 1):
        ts_new[2*i] = ts[i]
        ts_new[2*i + 1] = (ts[i] + ts[i + 1]) / 2
    ts_new[-1] = ts[-1]
    return ts_new

def stepSizedIntegrationRefinement(f, x0, ts, u, dt_min, dt_max, tol, max_iter=100):
    xk = x0
    uk = u[0]
    t_final = ts[-1]
    tk = ts[0]
    dt = ts[1] - ts[0]
    ts_new = [tk]
    while tk < t_final:
        xk, dt = rk45(f, xk, uk, tk, dt, dt_min, dt_max, tol=tol, t_final=t_final, max_iter=max_iter)
        tk += dt
        ts_new.append(float(tk))
        index = 0

        while index != len(u) - 1:
            if tk < (ts[index] + ts[index + 1]) / 2:
                break
            index += 1
        uk = u[index]
    return np.array(ts_new)


"""
Runge-Kutta of order 4-5 with step size control.
"""
def rk45(f, xk, uk, tk, dt, dt_min, dt_max, tol, t_final, alpha_min=0.25, alpha_max=1.75, beta=0.8, max_iter=100, *args, **kargs):
        q = 2
        iteration = 0
        while q > 1 and iteration < max_iter:
                k1 = f(xk, uk, tk)
                k2 = f(xk + dt * (1/5 * k1), uk, tk + dt * 1/5)
                k3 = f(xk + dt * (3/40 * k1 + 9 / 40 * k2), uk, tk + dt * 3/10)
                k4 = f(xk + dt * (44/55 * k1 - 56/15 * k2 + 32/9 * k3), uk, tk  + dt * 4/5)
                k5 = f(xk + dt * (19372 / 6561 * k1 - 25360 / 2187 * k2 + 64448 / 6561 * k3 - 212/729 * k4), uk, tk + dt * 8/9)
                k6 = f(xk + dt * (9017/3168 * k1 - 355/33 * k2 + 46732/5247 * k3 + 49/176 * k4 - 5103/18656 * k5), uk, tk + dt)
                k7 = f(xk + dt * (35/384 * k1 + 500/113 * k3 + 125/192 * k4 - 2187/6784 * k5 + 11/84 * k6), uk, tk + dt)
                y_6 = xk + dt * (35/384 * k1 + 500/1113 * k3 + 125/192 * k4 - 2187/6784 * k5 + 11/84 * k6) # Order 5
                y_7 = xk + dt * (5179/57600 * k1 + 7571/16695 * k3 + 393/640 * k4 -92097/339200 * k5 + 187/2100 * k6 + 1/40 * k7) # Order 4
                s = sqrt(sum((y_7 - y_6)**2))
                q = evalf(s / (dt * tol))
                alpha = min(max(alpha_min,q ** -1/5), alpha_max)
                dt = min(max(dt_min, beta*alpha*dt), dt_max) 
                if dt > t_final - tk:
                    dt = t_final - tk
                    break
                if dt == dt_min:
                        break 
                iteration += 1
        return y_6, dt