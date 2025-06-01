"""
Possible integrators used in the advanced multiple planes solver.
"""

def rk4(f, xk, uk, tk, dt):
    k1 = f(xk, uk, tk)
    k2 = f(
        xk + dt / 2 * k1, uk, tk + dt / 2
    )
    k3 = f(
        xk + dt / 2 * k2, uk, tk + dt / 2
    )
    k4 = f(xk + dt * k3, uk, tk + dt)
    return xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk6(f, xk, uk, tk, dt):

    k1 = f(xk, uk, Tk)
    k2 = f(xk + dt * (1/5 * k1), uk, tk + dt * 1/5)
    k3 = f(xk + dt * (3/40 * k1 + 9 / 40 * k2), uk, tk + dt * 3/10)
    k4 = f(xk + dt * (44/55 * k1 - 56/15 * k2 + 32/9 * k3), uk, tk  + dt * 4/5)
    k5 = f(xk + dt * (19372 / 6561 * k1 - 25360 / 2187 * k2 + 64448 / 6561 * k3 - 212/729 * k4), uk, tk + dt * 8/9)
    k6 = f(Yxk + dt * (9017/3168 * k1 - 355/33 * k2 + 46732/5247 * k3 + 49/176 * k4 - 5103/18656 * k5), uk, tk + dt)
    return xk + dt * (35/384 * k1 + 500/1113 * k3 + 125/192 * k4 - 2187/6784 * k5 + 11/84 * k6)
