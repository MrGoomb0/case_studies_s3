"""
Opti implementation to solve the multiple plane OCP.
"""

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from .wind_models import originalWindModel
from .integrators import rk4

# Time and discretization
tf = 40  # final time [sec]
N = 80  # number of control intervals
dt = tf / N  # time step

# Aircraft physical constants
m = 4662  # mass [lb sec^2 / ft]
g = 32.172  # gravity [ft/sec^2]
delta = 0.03491  # thrust inclination angle [rad]

# Thrust model coefficients: T = A0 + A1*V + A2*V^2
A0 = 0.4456e5  # [lb]
A1 = -0.2398e2  # [lb sec / ft]
A2 = 0.1442e-1  # [lb sec^2 / ft^2]

# Aerodynamic model
rho = 0.2203e-2  # air density [lb sec^2 / ft^4]
S = 0.1560e4  # reference surface area [ft^2]

# Wind model 3 beta (smoothing) parameters
beta0 = 0.3825  # initial beta value (approximate)
beta_dot0 = 0.2  # initial beta rate
sigma = 3  # time to reach beta = 1 [sec]

# C_D(alpha) = B0 + B1 * alpha + B2 * alpha**2, D = 0.5 * C_D(α) * ρ * S * V²
B0 = 0.1552
B1 = 0.12369  # [1/rad]
B2 = 2.4203  # [1/rad^2]

# Lift coefficient: C_L = C0 + C1 * alpha (+ C2 * alpha**2)
C0 = 0.7125  # baseline lift coefficient
C1 = 6.0877  # AOA lift slope [1/rad]

# Lift/drag model optional extensions (if needed)
C2 = -9.0277  # [rad^-2] — e.g., for moment or drag extension

# Angle of attack & control constraints
umax = 0.05236  # max control input (rate of change of alpha) [rad/sec]
alphamax = 0.3  # max angle of attack [rad]
alpha_star = 0.20944  # changing pt of AoA

# Wind model x parameters (piecewise smooth wind)
a = 6e-8  # x transition midpoint [ft]
b = -4e-11  # second transition point [ft]

# Wind model h parameters (polynomial form)
c = -np.log(25 / 30.6) * 1e-12  # transition smoothing width [ft]
d = -8.02881e-8  # polynomial coeff [sec^-1 ft^-2]
e = 6.28083e-11  # polynomial coeff [sec^-1 ft^-3]

# Cost function / target altitude
hR = 1000  # reference altitude [ft]
h_star = 1000  # used in some wind models

# Auxiliary
eps = 1e-6  # to avoid division by zero in V

# Scaling factors (used if normalizing states)
xscale = 10000  # [ft]
hscale = 1000  # [ft]
Vscale = 240  # [ft/sec]
gammascale = 0.1  # [rad]
alphascale = 0.3  # [rad]
uscale = 0.0523  # [rad/sec]


def solve_multiple_plain_ocp_advanced(k_values: np.array, init_estimate=None, h_final=850, windmodel=originalWindModel, integrator=rk4, verbose=True):

    M = len(k_values)  # number of parallel optimization problems
    if not verbose:
        print_level = 0
    else:
        print_level = 5
    # Opti instance and scaled variables
    opti = ca.Opti()
    x_s = opti.variable(M, N + 1)
    h_s = opti.variable(M, N + 1)
    V_s = opti.variable(M, N + 1)
    gamma_s = opti.variable(M, N + 1)
    alpha_s = opti.variable(M, N + 1)
    u_s = opti.variable(N)

    w = opti.variable()

    # Unscaled variables for dynamics
    x = x_s * xscale
    h = h_s * hscale
    V = V_s * Vscale
    gamma = gamma_s * gammascale
    alpha = alpha_s * alphascale
    u = u_s * uscale

    X_sym = ca.MX.sym("X", 5)
    u_sym = ca.MX.sym("u")

    opti.subject_to(x_s[:, 0] == 0)
    opti.subject_to(h_s[:, 0] == 600 / hscale)
    opti.subject_to(V_s[:, 0] == 239.7 / Vscale)
    opti.subject_to(V_s[:] >= 1e-2 / Vscale)
    opti.subject_to(gamma_s[:, 0] == -0.03925 / gammascale)
    opti.subject_to(alpha_s[:, 0] == min(0.1283, alphascale) / alphascale)
    opti.subject_to(gamma_s[:, -1] == 0.1296 / gammascale)
    opti.subject_to(
        h_s[:, -1] >= h_final / hscale
    )  # Extra constraint to get out of local minima.

    for j in range(M):
        for i in range(N):
            tk = i * dt  # New line
            Xk = ca.vertcat(x[j, i], h[j, i], V[j, i], gamma[j, i], alpha[j, i])
            Uk = u[i]
            f = aircraft_ode(windmodel, k_values[j])
            Xk_end = rk4(
                f,
                Xk,
                Uk,
                tk,
                dt,
            )
            X_next = ca.vertcat(
                x[j, i + 1],
                h[j, i + 1],
                V[j, i + 1],
                gamma[j, i + 1],
                alpha[j, i + 1],
            )
            opti.subject_to(X_next == Xk_end)
            opti.subject_to(opti.bounded(-1, u_s[i], 1))
            opti.subject_to(opti.bounded(-1, alpha_s[j, i], 1))

            opti.subject_to(w <= h_s[j, i])

    # Initial guess
    if init_estimate is None:
        for j in range(M):
            opti.set_initial(x_s[j, :], ca.linspace(0, 1, N + 1))
            opti.set_initial(h_s[j, :], 0.6)
            opti.set_initial(V_s[j, :], 239.7 / Vscale)
            opti.set_initial(gamma_s[j, :], -0.01 / gammascale)
            opti.set_initial(alpha_s[j, :], 0.02 / alphascale)
        opti.set_initial(u_s, 1)
        opti.set_initial(w, 0.5)
    else:
        opti.set_initial(x_s, init_estimate["x"])
        opti.set_initial(h_s, init_estimate["h"])
        opti.set_initial(V_s, init_estimate["V"])
        opti.set_initial(gamma_s, init_estimate["gamma"])
        opti.set_initial(alpha_s, init_estimate["alpha"])
        opti.set_initial(u_s, init_estimate["u"])
        opti.set_initial(w, 0.5)

    # Cost function
    opti.minimize(-w)

    # Solver
    opti.solver(
        "ipopt",
        {"expand": True},
        {
            "max_iter": 1000,
            "tol": 1e-6,
            "print_level": print_level,
            "linear_solver": "mumps",
            "hessian_approximation": "limited-memory",
        },
    )

    try:
        # opti.callback(lambda i: print("Alpha max:", np.max(opti.debug.value(alpha_s))))
        sol = opti.solve()
    except RuntimeError as e:
        opti.debug.show_infeasibilities()
        raise e

    return {
        "x": sol.value(x),
        "h": sol.value(h),
        "V": sol.value(V),
        "gamma": sol.value(gamma),
        "alpha": sol.value(alpha),
        "u": sol.value(u),
        "w": sol.value(w),
    }

def C_L(alpha_):
    return ca.if_else(
        alpha_ > alpha_star,
        C0 + C1 * alpha_,
        C0 + C1 * alpha_ + C2 * (alpha_ - alpha_star) ** 2,
    )


def beta(t_):
    return ca.if_else(t_ < sigma, beta0 + beta_dot0 * t_, 1.0)


def aircraft_ode(windmodel, k_value):
    x_ = ca.MX.sym('x')
    h_ = ca.MX.sym('h')
    V_ = ca.MX.sym('V')
    gamma_ = ca.MX.sym('gamma')
    alpha_ = ca.MX.sym('alpha')

    u_ = ca.MX.sym('u')
    t_ = ca.MX.sym('t')

    T = beta(t_) * (A0 + A1 * V_ + A2 * V_**2)
    D = 0.5 * (B0 + B1 * alpha_ + B2 * alpha_**2) * rho * S * V_**2
    L = 0.5 * rho * S * C_L(alpha_) * V_**2

    Wx, Wh = windmodel(x_, h_, k_value)
    dWx_dx_fun = ca.Function("dWx_dx", [x_], [ca.gradient(Wx, x_)])
    dWh_dx_fun = ca.Function("dWh_dx", [x_, h_], [ca.gradient(Wh, x_)])
    dWh_dh_fun = ca.Function("dWh_dh", [x_, h_], [ca.gradient(Wh, h_)])

    V_safe = ca.fmax(V_, 1e-3)

    x_dot = V_ * ca.cos(gamma_) + Wx
    h_dot = V_ * ca.sin(gamma_) + Wh

    dWx_dx_val = dWx_dx_fun(x_)[0]
    dWh_dx_val = dWh_dx_fun(x_, h_)[0]
    dWh_dh_val = dWh_dh_fun(x_, h_)[0]

    Wx_dot = dWx_dx_val * x_dot
    Wh_dot = dWh_dx_val * x_dot + dWh_dh_val * h_dot

    V_dot = (
        T / m * ca.cos(alpha_ + delta)
        - D / m
        - g * ca.sin(gamma_)
        - (Wx_dot * ca.cos(gamma_) + Wh_dot * ca.sin(gamma_))
    )
    gamma_dot = (
        T / (m * V_safe) * ca.sin(alpha_ + delta)
        + L / (m * V_safe)
        - g / V_safe * ca.cos(gamma_)
        + (1 / V_safe) * (Wx_dot * ca.sin(gamma_) - Wh_dot * ca.cos(gamma_))
    )
    alpha_dot = u_

    y0 = ca.vertcat(x_, h_, V_, gamma_, alpha_)
    yk = ca.vertcat(x_dot, h_dot, V_dot, gamma_dot, alpha_dot)
    return ca.Function('f', [y0, u_, t_], [yk])

# if __name__ == "__main__":
