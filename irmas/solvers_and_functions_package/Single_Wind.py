from casadi import *
from dataclasses import dataclass
from matplotlib import pyplot as plt
import numpy as np
import math
import scipy

@dataclass
class Parameters:
    # Time and discretization
    tf: float = 40          # final time [sec]
    nu: int = 80            # number of control intervals
    dut: float = tf / nu    # time step

    # Aircraft physical constants
    m: float = 4662                 # mass [lb sec^2 / ft]
    g: float = 32.172               # gravity [ft/sec^2]
    delta: float = 0.03491*pi/180   # thrust inclination angle [rad]

    # Thrust model coefficients: T = A0 + A1*V + A2*V^2
    A0: float = 0.4456e5    # [lb]
    A1: float = -0.2398e2   # [lb sec / ft]
    A2: float = 0.1442e-1   # [lb sec^2 / ft^2]

    # Aerodynamic model
    rho: float = 0.2203e-2  # air density [lb sec^2 / ft^4]
    S: float = 0.1560e4     # reference surface area [ft^2]

    # Wind model 3 beta (smoothing) parameters
    beta0: float = 0.3825                   # initial beta value (approximate)
    beta_dot0: float = 0.2                  # initial beta rate
    sigma: float = (1-beta0)/beta_dot0      # time to reach beta = 1 [sec]

    # C_D(alpha) = B0 + B1 * alpha + B2 * alpha**2, D = 0.5 * C_D(α) * ρ * S * V²
    B0: float = 0.1552
    B1: float = 0.12369     # [1/rad]
    B2: float = 2.4203      # [1/rad^2]

    # Lift coefficient: C_L = C0 + C1 * alpha (+ C2 * alpha**2)
    C0: float = 0.7125      # baseline lift coefficient
    C1: float = 6.0877      # AOA lift slope [1/rad]

    # Lift/drag model optional extensions (if needed)
    C2: float = -9.0277     # [rad^-2] — e.g., for moment or drag extension

    # Angle of attack & control constraints
    umax: float = 3*pi/180          # max control input (rate of change of alpha) [rad/sec]
    alphamax: float = 17.2*pi/180   # max angle of attack [rad]
    alpha_star: float = 12*pi/180   # changing pt of AoA

    # Wind model x parameters (piecewise smooth wind)
    a: float = 6e-8         # x transition midpoint [ft]
    b: float = -4e-11       # second transition point [ft]

    # Wind model h parameters (polynomial form)
    c: float = -np.log(25 / 30.6) * 1e-12       # transition smoothing width [ft]
    d: float = -8.02881e-8                      # polynomial coeff [sec^-1 ft^-2]
    e: float = 6.28083e-11                      # polynomial coeff [sec^-1 ft^-3]

    # Cost function / target altitude
    hR: float = 1000        # reference altitude [ft]
    h_star: float = 1000    # used in some wind models

    # Auxiliary
    eps: float = 1e-6       # to avoid division by zero in V

    # objective
    q: int = 6

    # scaling parameters
    scale_x = DM([1 / 10000, 1 / 1000, 1 / 100, 1 / 0.1, 1 / 0.1])
    inv_scale_x = DM([10000, 1000, 100, 0.1, 0.1])
    scale_u = 1 / 0.1
    inv_scale_u = 0.1
    scale_h = scale_x[1]
    inv_scale_h = inv_scale_x[1]
    scale_Q = 1/10**(17)
    inv_scale_Q = 10**(17)


params = Parameters()

# ---------- Wind models -------------

def Smooth(x1, x_start, x_end):
    t_smooth = (x1 - x_start) / (x_end - x_start + params.eps)
    return if_else(x1 < x_start, 0,
                   if_else(x1 > x_end, 1, 6 * t_smooth ** 5 - 15 * t_smooth ** 4 + 10 * t_smooth ** 3))

def A_wm1(x1, s):
    A1 = -50 + params.a * (x1 / s) ** 3 + params.b * (x1 / s) ** 4
    A2 = 0.025 * ((x1 / s) - 2300)
    A3 = 50 - params.a * (4600 - (x1 / s)) ** 3 - params.b * (4600 - (x1 / s)) ** 4
    A4 = 50
    return if_else(x1 <= 500 * s, A1,
                   if_else(x1 <= 4100 * s, A2,
                           if_else(x1 <= 4600 * s, A3, A4)))

def B_wm1(x1, s):
    B1 = params.d * (x1 / s) ** 3 + params.e * (x1 / s) ** 4
    B2 = -51 * exp(fmin(-params.c * ((x1 / s) - 2300) ** 4, 30))
    B3 = params.d * (4600 - (x1 / s)) ** 3 + params.e * (4600 - (x1 / s)) ** 4
    B4 = 0
    return if_else(x1 <= 500 * s, B1,
                   if_else(x1 <= 4100 * s, B2,
                           if_else(x1 <= 4600 * s, B3, B4)))

def A_wm1s(x1, s):
    A1 = -50 + params.a * (x1 / s) ** 3 + params.b * (x1 / s) ** 4
    A2 = 0.025 * ((x1 / s) - 2300)
    A3 = 50 - params.a * (4600 - (x1 / s)) ** 3 - params.b * (4600 - (x1 / s)) ** 4
    A4 = 50
    s1 = Smooth(x1, 480 * s, 520 * s)
    s2 = Smooth(x1, 4080 * s, 4120 * s)
    s3 = Smooth(x1, 4580 * s, 4620 * s)
    B12 = (1 - s1) * A1 + s1 * A2
    B23 = (1 - s2) * A2 + s2 * A3
    B34 = (1 - s3) * A3 + s3 * A4
    return if_else(x1 <= 500 * s, B12,
                   if_else(x1 <= 4100 * s, B23,
                           if_else(x1 <= 4600 * s, B34, A4)))

def B_wm1s(x1, s):
    B1 = params.d * (x1 / s) ** 3 + params.e * (x1 / s) ** 4
    B2 = -51 * exp(fmin(-params.c * ((x1 / s) - 2300) ** 4, 30))
    B3 = params.d * (4600 - (x1 / s)) ** 3 + params.e * (4600 - (x1 / s)) ** 4
    B4 = 0
    s1 = Smooth(x1, 480 * s, 520 * s)
    s2 = Smooth(x1, 4080 * s, 4120 * s)
    s3 = Smooth(x1, 4580 * s, 4620 * s)
    B12 = (1 - s1) * B1 + s1 * B2
    B23 = (1 - s2) * B2 + s2 * B3
    B34 = (1 - s3) * B3 + s3 * B4
    return if_else(x1 <= 500 * s, B12,
                   if_else(x1 <= 4100 * s, B23,
                           if_else(x1 <= 4600 * s, B34, B4)))

def A_wm2(x1, s):
    m = 1/40
    a = 2000
    c = 2300 * s
    d = 150
    return 50*d / (2*s*a) * (
        log(cosh( ((x1-(c)) + s*a) / d))
        - log(cosh( ((x1-(c)) - s*a) / d))
        )

def B_wm2(x1, s):
    a = 1150**2
    c = 1350**2
    d = 630**2
    l = 2300 * s
    return -25.5 + 25.5*(d*s)  /(2*s*a) * (
        log(cosh( ((x1 - l)**2 - s**2*c + s**2*a) / (d*s**2)))
        - log(cosh( ((x1 - l)**2 - s**2*c - s**2*a) / (d*s**2)))
        )

def plot_wind(k: float, s:float, model: int, smooth: bool):
    X = np.arange(0, 10001, 1)
    if model == 1:
        if model is not smooth:
            A = np.vectorize(A_wm1)
            B = np.vectorize(B_wm1)
        else:
            A = np.vectorize(A_wm1s)
            B = np.vectorize(B_wm1s)

    if model == 2:
        A = np.vectorize(A_wm2)
        B = np.vectorize(B_wm2)

    # Plot A(x) and B(x)
    plt.figure(figsize=(12, 6))
    plt.plot(X, A(X, s), '--', label='A(x)')
    plt.plot(X, B(X, s), '-', label='B(x)')
    plt.title('Wind Functions A(x) and B(x)')
    plt.xlabel('x [ft]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Wind streamplot using smooth A and B
    x_grid = np.linspace(0, 10000, 200)
    h_grid = np.linspace(0, 2000, 100)
    X, H = np.meshgrid(x_grid, h_grid)

    U = k * A(X, s)
    V = k * H * B(X, s) / params.h_star

    plt.figure(figsize=(12, 6))
    plt.streamplot(X, H, U, V, color='black', linewidth=1, density=1.2, arrowsize=0.7)
    plt.title('Wind Vector Field')
    plt.xlabel('x [ft]')
    plt.ylabel('h [ft]')
    plt.xlim([0, 10000])
    plt.ylim([0, 2000])
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# ---------- Integrators ----------
def rk1_step_bolza(f, qf, xk, uk, tk, dt, qk):
    return xk + dt * f(xk, uk, tk), qk + dt * qf(xk)

def rk2_step_bolza(f, qf, xk, uk, tk, dt, qk):
    k1 = f(xk, uk, tk)
    k2 = f(xk + dt * k1, uk, tk + dt)
    qk1 = qf(xk)
    qk2 = qf(xk + dt * k1)
    return xk + dt / 2 * (k1 + k2), qk + dt / 2 * (qk1 + qk2)

def rk4_step_bolza(f, qf, xk, uk, tk, dt, qk):
    k1 = f(xk, uk, tk)
    k2 = f(xk + dt / 2 * k1, uk, tk + dt / 2)
    k3 = f(xk + dt / 2 * k2, uk, tk + dt / 2)
    k4 = f(xk + dt * k3, uk, tk + dt)
    qk1 = qf(xk)
    qk2 = qf(xk + dt / 2 * k1)
    qk3 = qf(xk + dt / 2 * k2)
    qk4 = qf(xk + dt * k3)
    return xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), qk + dt / 6 * (qk1 + 2 * qk2 + 2 * qk3 + qk4)

def rk6_step_bolza(f, qf, xk, uk, tk, dt, qk):
    k1 = f(xk, uk, tk)
    k2 = f(xk + dt / 3 * k1, uk, tk + dt / 3)
    k3 = f(xk + dt / 6 * k1 + dt / 6 * k2, uk, tk + dt / 3)
    k4 = f(xk + dt / 8 * k1 + 3 * dt / 8 * k3, uk, tk + dt / 2)
    k5 = f(xk + dt * (0.5 * k1 - 1.5 * k3 + 2 * k4), uk, tk + dt)
    k6 = f(xk + dt * (-1.5 * k1 + 2 * k2 + 1.5 * k3 - 2 * k4 + 1.5 * k5), uk, tk + dt)
    k7 = f(xk + dt * (3 / 7 * k1 + 6 / 7 * k4 - 12 / 7 * k5 + 8 / 7 * k6), uk, tk + dt)
    q1 = qf(xk)
    q2 = qf(xk + dt / 3 * k1)
    q3 = qf(xk + dt / 6 * k1 + dt / 6 * k2)
    q4 = qf(xk + dt / 8 * k1 + 3 * dt / 8 * k3)
    q5 = qf(xk + dt * (0.5 * k1 - 1.5 * k3 + 2 * k4))
    q6 = qf(xk + dt * (-1.5 * k1 + 2 * k2 + 1.5 * k3 - 2 * k4 + 1.5 * k5))
    q7 = qf(xk + dt * (3 / 7 * k1 + 6 / 7 * k4 - 12 / 7 * k5 + 8 / 7 * k6))
    return xk + dt * ((7 / 90) * k1 + (16 / 45) * k3 + (2 / 15) * k4 + (16 / 45) * k5 + (7 / 90) * k7), qk + dt * ((7 / 90) * q1 + (16 / 45) * q3 + (2 / 15) * q4 + (16 / 45) * q5 + (7 / 90) * q7)

def rk1_step(f, xk, uk, tk, dt):
    return xk + dt * f(xk, uk, tk)

def rk2_step(f, xk, uk, tk, dt):
    k1 = f(xk, uk, tk)
    k2 = f(xk + dt * k1, uk, tk + dt)
    return xk + dt / 2 * (k1 + k2)

def rk4_step(f, xk, uk, tk, dt):
    k1 = f(xk, uk, tk)
    k2 = f(xk + dt / 2 * k1, uk, tk + dt / 2)
    k3 = f(xk + dt / 2 * k2, uk, tk + dt / 2)
    k4 = f(xk + dt * k3, uk, tk + dt)
    return xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def rk6_step(f, xk, uk, tk, dt):
    k1 = f(xk, uk, tk)
    k2 = f(xk + dt / 3 * k1, uk, tk + dt / 3)
    k3 = f(xk + dt / 6 * k1 + dt / 6 * k2, uk, tk + dt / 3)
    k4 = f(xk + dt / 8 * k1 + 3 * dt / 8 * k3, uk, tk + dt / 2)
    k5 = f(xk + dt * (0.5 * k1 - 1.5 * k3 + 2 * k4), uk, tk + dt)
    k6 = f(xk + dt * (-1.5 * k1 + 2 * k2 + 1.5 * k3 - 2 * k4 + 1.5 * k5), uk, tk + dt)
    k7 = f(xk + dt * (3 / 7 * k1 + 6 / 7 * k4 - 12 / 7 * k5 + 8 / 7 * k6), uk, tk + dt)
    return xk + dt * ((7 / 90) * k1 + (16 / 45) * k3 + (2 / 15) * k4 + (16 / 45) * k5 + (7 / 90) * k7)

def MC_rk2_step(dynamics, xk, uk, tk, dt, k_value, s_value):
    k1, _ = dynamics(xk, uk, tk, k_value, s_value)
    k2, x2dotdot = dynamics(xk + dt * k1, uk, tk + dt, k_value, s_value)
    return xk + dt / 2 * (k1 + k2), x2dotdot

def MC_rk4_step(dynamics, xk, uk, tk, dt, k_value, s_value):
    k1, _ = dynamics(xk, uk, tk, k_value, s_value)
    k2, _ = dynamics(xk + dt / 2 * k1, uk, tk + dt / 2, k_value, s_value)
    k3, _ = dynamics(xk + dt / 2 * k2, uk, tk + dt / 2, k_value, s_value)
    k4, x2dotdot = dynamics(xk + dt * k3, uk, tk + dt, k_value, s_value)
    return xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), x2dotdot



# ---------- Bolza -----------
def solver_bolza(k_value, s_value, A_w, B_w, N: int = params.nu, x_initial=None, pesch_end_cond: bool=False, integrator = rk4_step_bolza, tol=1e-10,constr_viol_tol=1e-6):
    params.nu = N
    params.dut = params.tf / params.nu

    # State, Time and Control
    x1 = MX.sym('x1')  # x
    x2 = MX.sym('x2')  # h
    x3 = MX.sym('x3')  # V
    x4 = MX.sym('x4')  # gamma
    x5 = MX.sym('x5')  # alpha
    t = MX.sym('t')  # time
    u = MX.sym('u')  # control
    x = vertcat(x1, x2, x3, x4, x5)

    # Wind
    wind_x_expr = k_value * A_w(x1, s_value)
    wind_x = Function('wind_x', [x1], [wind_x_expr])
    wind_h_expr = k_value * x2 * B_w(x1, s_value) / params.h_star
    wind_h = Function('wind_h', [x1, x2], [wind_h_expr])

    dWx_dx = Function("dWx_dx", [x1], [gradient(wind_x_expr, x1)])
    dWh_dx = Function("dWh_dx", [x1, x2], [gradient(wind_h_expr, x1)])
    dWh_dh = Function("dWh_dh", [x1, x2], [gradient(wind_h_expr, x2)])

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2

    # ode
    x1dot = x3 * cos(x4) + wind_x(x1)
    x2dot = x3 * sin(x4) + wind_h(x1, x2)

    wxdot = dWx_dx(x1) * x1dot
    whdot = dWh_dx(x1, x2) * x1dot + dWh_dh(x1, x2) * x2dot
    x3_safe = fmax(x3, params.eps)

    x3dot = T / params.m * cos(x5 + params.delta) - D / params.m - params.g * sin(x4) - (
                wxdot * cos(x4) + whdot * sin(x4))
    x4dot = T / (params.m * x3) * sin(x5 + params.delta) + L / (params.m * x3) - params.g / x3_safe * cos(x4) + (
                1 / x3_safe) * (wxdot * sin(x4) - whdot * cos(x4))
    x5dot = u

    f = Function('f', [x, u, t], [vertcat(x1dot, x2dot, x3dot, x4dot, x5dot)])

    x2dotdot_expr = x3dot * sin(x4) + x3 * x4dot * cos(x4) + whdot
    x2dotdot = Function('x2dotdot', [x, u, t], [x2dotdot_expr])

    # objective function
    Q = (params.hR - x2) ** params.q
    qf = Function('qf', [x], [Q])

    # npl prep
    w = []  # List of decision variables
    w0 = []  # Initial guess
    lbw = []  # Lower bounds
    ubw = []  # Upper bounds
    g = []  # Constraints
    lbg = []  # Lower bound on constraints
    ubg = []  # Upper bound on constraints
    J = 0  # Objective function

    # initial condition
    if x_initial is None:
        x_initial = [0, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
    Xk = MX.sym('X_0', 5)
    w += [Xk]
    lbw += x_initial
    ubw += x_initial
    w0 += x_initial

    Tk = 0

    # npl set up
    for k in range(params.nu):
        # control
        Uk = MX.sym('U_' + str(k))
        w += [Uk]
        lbw += [-3 * pi / 180]
        ubw += [3 * pi / 180]
        w0 += [0]

        # state integration
        Xk_end, J_end = integrator(f, qf, Xk, Uk, Tk, params.dut, J)
        J = J_end
        h_ddot_val = x2dotdot(Xk, Uk, Tk)

        Tk += params.dut

        # state
        Xk = MX.sym('X_' + str(k + 1), 5)
        w += [Xk]
        lbw += [-inf, 0, -inf, -inf, -17.2 * pi / 180]
        ubw += [inf, params.hR, inf, inf, 17.2 * pi / 180]
        w0 += [k * 10000 / params.nu, x_initial[1], x_initial[2], x_initial[3], x_initial[4]]
        g += [Xk_end - Xk]
        lbg += [0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0]

        # Add to constraint list
        g += [h_ddot_val]
        lbg += [-2 * params.g]
        ubg += [10 * params.g]

    if pesch_end_cond:
        XF = 7.431 * pi / 180
        g += [Xk[3] - XF]
        lbg += [0]
        ubg += [0]
    else:
        g += [Xk[3]]
        lbg += [0]
        ubg += [inf]

    opts = {'ipopt': {'print_level': 3, 'tol': tol, 'constr_viol_tol': constr_viol_tol}}

    npl = {'x': vertcat(*w), 'f': J, 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', npl, opts)

    arg = {'x0': w0, 'lbx': lbw, 'ubx': ubw, 'lbg': lbg, 'ubg': ubg}

    # Solve
    sol = solver(**arg)
    w_opt = sol['x'].full().flatten()
    J_opt = sol['f'].full().flatten()

    return w_opt, J_opt

def solver_bolza_scaled(k_value, s_value, A_w, B_w, N: int = params.nu, x_initial=None, pesch_end_cond: bool=False, integrator = rk4_step_bolza, tol=1e-10,constr_viol_tol=1e-6,q=params.q,scale_Q=params.scale_Q):
    params.nu = N
    params.dut = params.tf / params.nu
    params.q=q
    params.scale_Q = scale_Q
    params.inv_scale_Q = 1/scale_Q

    # State, Time and Control
    x1 = MX.sym('x1')  # x
    x2 = MX.sym('x2')  # h
    x3 = MX.sym('x3')  # V
    x4 = MX.sym('x4')  # gamma
    x5 = MX.sym('x5')  # alpha
    t = MX.sym('t')  # time
    u = MX.sym('u')  # control
    x = vertcat(x1, x2, x3, x4, x5)

    # Wind
    wind_x_expr = k_value * A_w(x1, s_value)
    wind_x = Function('wind_x', [x1], [wind_x_expr])
    wind_h_expr = k_value * x2 * B_w(x1, s_value) / params.h_star
    wind_h = Function('wind_h', [x1, x2], [wind_h_expr])

    dWx_dx = Function("dWx_dx", [x1], [gradient(wind_x_expr, x1)])
    dWh_dx = Function("dWh_dx", [x1, x2], [gradient(wind_h_expr, x1)])
    dWh_dh = Function("dWh_dh", [x1, x2], [gradient(wind_h_expr, x2)])

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2

    # ode
    x1dot = x3 * cos(x4) + wind_x(x1)
    x2dot = x3 * sin(x4) + wind_h(x1, x2)

    wxdot = dWx_dx(x1) * x1dot
    whdot = dWh_dx(x1, x2) * x1dot + dWh_dh(x1, x2) * x2dot
    x3_safe = fmax(x3, params.eps)

    x3dot = T / params.m * cos(x5 + params.delta) - D / params.m - params.g * sin(x4) - (
                wxdot * cos(x4) + whdot * sin(x4))
    x4dot = T / (params.m * x3) * sin(x5 + params.delta) + L / (params.m * x3) - params.g / x3_safe * cos(x4) + (
                1 / x3_safe) * (wxdot * sin(x4) - whdot * cos(x4))
    x5dot = u

    f_org = Function('f_org', [x, u, t], [vertcat(x1dot, x2dot, x3dot, x4dot, x5dot)])

    x2dotdot_expr = x3dot * sin(x4) + x3 * x4dot * cos(x4) + whdot
    x2dotdot = Function('x2dotdot', [x, u, t], [x2dotdot_expr])

    # scaled state, control and sys dynamics
    xs = MX.sym('xs', 5)
    us = MX.sym('us')
    f_s = params.scale_x * f_org(params.inv_scale_x * xs, params.inv_scale_u * us, t)
    f_scaled = Function('f_scaled', [xs, us, t], [f_s])

    # objective function
    Q = (params.hR - x2) ** params.q
    qf_org = Function('qf', [x], [Q])
    qf_s = params.scale_Q*qf_org(params.inv_scale_x * xs)
    qf_scaled = Function('qf_scaled', [xs], [qf_s])

    # npl prep
    w = []  # List of decision variables
    w0 = []  # Initial guess
    lbw = []  # Lower bounds
    ubw = []  # Upper bounds
    g = []  # Constraints
    lbg = []  # Lower bound on constraints
    ubg = []  # Upper bound on constraints
    Js = 0  # Objective function

    # initial condition
    if x_initial is None:
        x_initial = [0, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
    xs_initial = [x_initial[i] * float(params.scale_x[i]) for i in range(5)]
    Xsk = MX.sym('Xs_0', 5)
    w += [Xsk]
    lbw += xs_initial
    ubw += xs_initial
    w0 += xs_initial

    Tk = 0

    # npl set up
    for k in range(params.nu):
        # control
        Usk = MX.sym('Us_' + str(k))
        w += [Usk]
        lbw += [-3 * pi / 180 * params.scale_u]
        ubw += [3 * pi / 180 * params.scale_u]
        w0 += [0]

        # state integration
        Xsk_end, Js_end = integrator(f_scaled,qf_scaled, Xsk, Usk, Tk, params.dut, Js)
        Js = Js_end
        h_ddot_val = x2dotdot(Xsk * params.inv_scale_x, Usk * params.inv_scale_u, Tk)

        Tk += params.dut

        # state
        Xsk = MX.sym('Xs_' + str(k + 1), 5)
        w += [Xsk]
        lbw += [-inf, 0, -inf, -inf, -17.2 * pi / 180 * params.scale_x[4]]
        ubw += [inf, inf, inf, inf, 17.2 * pi / 180 * params.scale_x[4]]
        w0 += [k * 10000 / params.nu * params.scale_x[0], 600 * params.scale_x[1], 239.7 * params.scale_x[2],
               -2.249 * pi / 180 * params.scale_x[3], 7.353 * pi / 180 * params.scale_x[4]]
        g += [Xsk_end - Xsk]
        lbg += [0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0]

        # Add to constraint list
        g += [h_ddot_val]
        lbg += [-2 * params.g]
        ubg += [10 * params.g]

    if pesch_end_cond:
        XsF = 7.431 * pi / 180 * float(params.scale_x[3])
        g += [Xsk[3] - XsF]
        lbg += [0]
        ubg += [0]
    else:
        g += [Xsk[3]]
        lbg += [0]
        ubg += [inf]


    opts = {'ipopt': {'print_level': 3, 'tol': tol, 'constr_viol_tol': constr_viol_tol}}

    npl = {'x': vertcat(*w), 'f': Js, 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', npl, opts)

    arg = {'x0': vertcat(*w0), 'lbx': vertcat(*lbw), 'ubx': vertcat(*ubw), 'lbg': vertcat(*lbg), 'ubg': vertcat(*ubg)}

    # Solve
    sol = solver(**arg)
    w_opt = sol['x'].full().flatten()
    J_opt = sol['f'].full().flatten()

    return w_opt, J_opt



# ---------- min -w -----------
def solver_min_h(k_value, s_value, A_w, B_w, N: int = params.nu, x_initial=None, pesch_end_cond: bool=False, integrator = rk4_step, tol=1e-10, constr_viol_tol=1e-6):
    params.nu = N
    params.dut = params.tf / params.nu

    # State, Time and Control
    x1 = MX.sym('x1')  # x
    x2 = MX.sym('x2')  # h
    x3 = MX.sym('x3')  # V
    x4 = MX.sym('x4')  # gamma
    x5 = MX.sym('x5')  # alpha
    t = MX.sym('t')  # time
    u = MX.sym('u')  # control
    x = vertcat(x1, x2, x3, x4, x5)

    # Wind
    wind_x_expr = k_value * A_w(x1,s_value)
    wind_x = Function('wind_x', [x1], [wind_x_expr])
    wind_h_expr = k_value * x2 * B_w(x1,s_value) / params.h_star
    wind_h = Function('wind_h', [x1, x2], [wind_h_expr])

    dWx_dx = Function("dWx_dx", [x1], [gradient(wind_x_expr, x1)])
    dWh_dx = Function("dWh_dx", [x1, x2], [gradient(wind_h_expr, x1)])
    dWh_dh = Function("dWh_dh", [x1, x2], [gradient(wind_h_expr, x2)])

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2

    # ode
    x1dot = x3 * cos(x4) + wind_x(x1)
    x2dot = x3 * sin(x4) + wind_h(x1, x2)

    wxdot = dWx_dx(x1) * x1dot
    whdot = dWh_dx(x1, x2) * x1dot + dWh_dh(x1, x2) * x2dot
    x3_safe = fmax(x3, params.eps)

    x3dot = T / params.m * cos(x5 + params.delta) - D / params.m - params.g * sin(x4) - (
            wxdot * cos(x4) + whdot * sin(x4))
    x4dot = T / (params.m * x3) * sin(x5 + params.delta) + L / (params.m * x3) - params.g / x3_safe * cos(x4) + (
            1 / x3_safe) * (wxdot * sin(x4) - whdot * cos(x4))
    x5dot = u

    f = Function('f', [x, u, t], [vertcat(x1dot, x2dot, x3dot, x4dot, x5dot)])

    x2dotdot_expr = x3dot * sin(x4) + x3 * x4dot * cos(x4) + whdot
    x2dotdot = Function('x2dotdot', [x, u, t], [x2dotdot_expr])

    # npl prep
    w = []  # List of decision variables
    w0 = []  # Initial guess
    lbw = []  # Lower bounds
    ubw = []  # Upper bounds
    g = []  # Constraints
    lbg = []  # Lower bound on constraints
    ubg = []  # Upper bound on constraints
    J = 0  # Objective function

    # initial condition
    min_h = MX.sym('min_h')
    w += [min_h]
    lbw += [0]
    ubw += [inf]
    w0 += [600]

    if x_initial is None:
        x_initial = [0, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
    Xk = MX.sym('X_0', 5)
    w += [Xk]
    lbw += x_initial
    ubw += x_initial
    w0 += x_initial

    Tk = 0

    # npl set up
    for k in range(params.nu):
        # control
        Uk = MX.sym('U_' + str(k))
        w += [Uk]
        lbw += [-3 * pi / 180]
        ubw += [3 * pi / 180]
        w0 += [0]

        # state integration
        Xk_end = integrator(f, Xk, Uk, Tk, params.dut)
        h_ddot_val = x2dotdot(Xk, Uk, Tk)

        Tk += params.dut

        # state
        Xk = MX.sym('X_' + str(k + 1), 5)
        w += [Xk]
        lbw += [-inf, 0, -inf, -inf, -17.2 * pi / 180]
        ubw += [inf, inf, inf, inf, 17.2 * pi / 180]
        w0 += [k * 10000 / params.nu, x_initial[1], x_initial[2], x_initial[3], x_initial[4]]
        g += [Xk_end - Xk]
        lbg += [0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0]

        g += [Xk[1] - min_h]
        lbg += [0]
        ubg += [inf]

        # Add to constraint list
        g += [h_ddot_val]
        lbg += [-2 * params.g]
        ubg += [10 * params.g]

    if pesch_end_cond:
        XF = 7.431 * pi / 180
        g += [Xk[3] - XF]
        lbg += [0]
        ubg += [0]
    else:
        g += [Xk[3]]
        lbg += [0]
        ubg += [inf]

    J = -min_h

    opts = {'ipopt': {'print_level': 3, 'tol': tol, 'constr_viol_tol': constr_viol_tol}}
    npl = {'x': vertcat(*w), 'f': J, 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', npl, opts)

    arg = {'x0': w0, 'lbx': lbw, 'ubx': ubw, 'lbg': lbg, 'ubg': ubg}

    # Solve
    sol = solver(**arg)
    w_opt = sol['x'].full().flatten()
    J_opt = sol['f'].full().flatten()
    return w_opt, J_opt

def solver_min_h_scaled(k_value, s_value, A_w, B_w, N: int = params.nu, x_initial=None, pesch_end_cond: bool=False, integrator = rk4_step, tol=1e-10, constr_viol_tol=1e-6):
    params.nu = N
    params.dut = params.tf / params.nu

    # State, Time and Control
    x1 = MX.sym('x1')  # x
    x2 = MX.sym('x2')  # h
    x3 = MX.sym('x3')  # V
    x4 = MX.sym('x4')  # gamma
    x5 = MX.sym('x5')  # alpha
    t = MX.sym('t')  # time
    u = MX.sym('u')  # control
    x = vertcat(x1, x2, x3, x4, x5)

    # Wind
    wind_x_expr = k_value * A_w(x1,s_value)
    wind_x = Function('wind_x', [x1], [wind_x_expr])
    wind_h_expr = k_value * x2 * B_w(x1,s_value) / params.h_star
    wind_h = Function('wind_h', [x1, x2], [wind_h_expr])

    dWx_dx = Function("dWx_dx", [x1], [gradient(wind_x_expr, x1)])
    dWh_dx = Function("dWh_dx", [x1, x2], [gradient(wind_h_expr, x1)])
    dWh_dh = Function("dWh_dh", [x1, x2], [gradient(wind_h_expr, x2)])

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2

    # ode
    x1dot = x3 * cos(x4) + wind_x(x1)
    x2dot = x3 * sin(x4) + wind_h(x1, x2)

    wxdot = dWx_dx(x1) * x1dot
    whdot = dWh_dx(x1, x2) * x1dot + dWh_dh(x1, x2) * x2dot
    x3_safe = fmax(x3, params.eps)

    x3dot = T / params.m * cos(x5 + params.delta) - D / params.m - params.g * sin(x4) - (
            wxdot * cos(x4) + whdot * sin(x4))
    x4dot = T / (params.m * x3) * sin(x5 + params.delta) + L / (params.m * x3) - params.g / x3_safe * cos(x4) + (
            1 / x3_safe) * (wxdot * sin(x4) - whdot * cos(x4))
    x5dot = u

    f_org = Function('f_org', [x, u, t], [vertcat(x1dot, x2dot, x3dot, x4dot, x5dot)])

    x2dotdot_expr = x3dot * sin(x4) + x3 * x4dot * cos(x4) + whdot
    x2dotdot = Function('x2dotdot', [x, u, t], [x2dotdot_expr])

    # scaled state, control and sys dynamics
    xs = MX.sym('xs', 5)
    us = MX.sym('us')
    f_s = params.scale_x * f_org(params.inv_scale_x * xs, params.inv_scale_u * us, t)
    f_scaled = Function('f_scaled', [xs, us, t], [f_s])

    # npl prep
    w = []  # List of decision variables
    w0 = []  # Initial guess
    lbw = []  # Lower bounds
    ubw = []  # Upper bounds
    g = []  # Constraints
    lbg = []  # Lower bound on constraints
    ubg = []  # Upper bound on constraints
    J = 0  # Objective function

    # initial condition
    min_h_s = MX.sym('min_h_s')
    w += [min_h_s]
    lbw += [0]
    ubw += [inf]
    w0 += [600*float(params.scale_h)]

    if x_initial is None:
        x_initial = [0, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
    xs_initial = [x_initial[i] * float(params.scale_x[i]) for i in range(5)]
    Xsk = MX.sym('Xs_0', 5)
    w += [Xsk]
    lbw += xs_initial
    ubw += xs_initial
    w0 += xs_initial

    Tk = 0

    # npl set up
    for k in range(params.nu):
        # control
        Usk = MX.sym('Us_' + str(k))
        w += [Usk]
        lbw += [-3 * pi / 180 * params.scale_u]
        ubw += [3 * pi / 180 * params.scale_u]
        w0 += [0]

        # state integration
        Xsk_end = integrator(f_scaled, Xsk, Usk, Tk, params.dut)
        h_ddot_val = x2dotdot(Xsk*params.inv_scale_x, Usk*params.inv_scale_u, Tk)

        Tk += params.dut

        # state
        Xsk = MX.sym('Xs_' + str(k + 1), 5)
        w += [Xsk]
        lbw += [-inf, 0, -inf, -inf, -17.2 * pi / 180 * params.scale_x[4]]
        ubw += [inf, inf, inf, inf, 17.2 * pi / 180 * params.scale_x[4]]
        w0 += [k * 10000 / params.nu * params.scale_x[0], 600 * params.scale_x[1], 239.7* params.scale_x[2], -2.249 * pi / 180 * params.scale_x[3], 7.353 * pi / 180 * params.scale_x[4]]
        g += [Xsk_end - Xsk]
        lbg += [0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0]

        g += [Xsk[1] - min_h_s]
        lbg += [0]
        ubg += [inf]

        # Add to constraint list
        g += [h_ddot_val]
        lbg += [-2 * params.g]
        ubg += [10 * params.g]
    if pesch_end_cond:
        XsF = 7.431 * pi / 180 * float(params.scale_x[3])
        g += [Xsk[3] - XsF]
        lbg += [0]
        ubg += [0]
    else:
        g += [Xsk[3]]
        lbg += [0]
        ubg += [inf]


    J = -min_h_s

    opts = {'ipopt': {'print_level': 3, 'tol': tol, 'constr_viol_tol': constr_viol_tol}}
    npl = {'x': vertcat(*w), 'f': J, 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', npl, opts)

    arg = {'x0': vertcat(*w0), 'lbx': vertcat(*lbw), 'ubx': vertcat(*ubw), 'lbg': vertcat(*lbg), 'ubg': vertcat(*ubg)}

    # Solve
    sol = solver(**arg)
    w_opt = sol['x'].full().flatten()
    J_opt = sol['f'].full().flatten()
    return w_opt, J_opt



# ------------ Ploting -------------

def ploter(w_opt: list[float], is_bolza: bool, is_scaled: bool):
    if is_bolza is False:
        w_opt = w_opt[1:]
    x1_opt = w_opt[0::6]  # start at index 0, get every 6th:
    x2_opt = w_opt[1::6]
    x3_opt = w_opt[2::6]
    x4_opt = w_opt[3::6]
    x5_opt = w_opt[4::6]
    u_opt = w_opt[5::6]

    if is_scaled:
        x1_opt = [x * float(params.inv_scale_x[0]) for x in x1_opt]
        x2_opt = [x * float(params.inv_scale_x[1]) for x in x2_opt]
        x3_opt = [x * float(params.inv_scale_x[2]) for x in x3_opt]
        x4_opt = [x * float(params.inv_scale_x[3]) for x in x4_opt]
        x5_opt = [x * float(params.inv_scale_x[4]) for x in x5_opt]
        u_opt = [x * float(params.inv_scale_u) for x in u_opt]

    N = params.nu
    tf = params.tf
    tgrid = [tf / N * k for k in range(N + 1)]

    print(min(x2_opt))

    plt.figure(figsize=(12, 6))
    plt.clf()
    plt.plot(x1_opt, x2_opt, '--')
    plt.xlabel('horizontal distance [ft]')
    plt.ylabel('altitude [ft]')
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.clf()
    plt.step(tgrid[:-1], u_opt, where='post')
    plt.xlabel('t')
    plt.grid()
    plt.show()
    return

def ploter_with_wind(w_opt: list[float], is_bolza: bool, is_scaled: bool, k: float, s:float, model: int, smooth: bool):
    if model == 1:
        if model is not smooth:
            A = np.vectorize(A_wm1)
            B = np.vectorize(B_wm1)
        else:
            A = np.vectorize(A_wm1s)
            B = np.vectorize(B_wm1s)

    if model == 2:
        if model is not smooth:
            A = np.vectorize(A_wm2)
            B = np.vectorize(B_wm2)


    # Wind field plot using streamplot (to look like dotted flow lines)
    x_grid = np.linspace(0, 10000, 200)
    h_grid = np.linspace(0, 1500, 100)
    X, H = np.meshgrid(x_grid, h_grid)

    U = k * A(X,s)
    V = k * H * B(X,s) / params.h_star



    if is_bolza is False:
        w_opt = w_opt[1:]
    x1_opt = w_opt[0::6]  # start at index 0, get every 6th:
    x2_opt = w_opt[1::6]
    x3_opt = w_opt[2::6]
    x4_opt = w_opt[3::6]
    x5_opt = w_opt[4::6]
    u_opt = w_opt[5::6]

    if is_scaled:
        x1_opt = [x * float(params.inv_scale_x[0]) for x in x1_opt]
        x2_opt = [x * float(params.inv_scale_x[1]) for x in x2_opt]
        x3_opt = [x * float(params.inv_scale_x[2]) for x in x3_opt]
        x4_opt = [x * float(params.inv_scale_x[3]) for x in x4_opt]
        x5_opt = [x * float(params.inv_scale_x[4]) for x in x5_opt]
        u_opt = [x * float(params.inv_scale_u) for x in u_opt]

    N = params.nu
    tf = params.tf
    tgrid = [tf / N * k for k in range(N + 1)]

    print(min(x2_opt))

    plt.figure(figsize=(12, 6))
    plt.streamplot(
        X, H, U, V,
        color='black', linewidth=1, density=1.2, arrowsize=0.1
    )
    plt.plot(x1_opt, x2_opt, '--')
    plt.xlabel('horizontal distance [ft]')
    plt.ylabel('altitude [ft]')
    plt.title('Optimal flight Path')
    plt.tight_layout()
    plt.xlim([0, 10000])
    plt.ylim([0, 1000])
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.clf()
    plt.step(tgrid[:-1], u_opt, where='post')
    plt.xlabel('t')
    plt.title('Optimal control')
    plt.grid()
    plt.xlim([0, 40])
    plt.show()
    return



# -------- Others --------------
def u_opt_return(w_opt: list[float], is_bolza: bool, is_scaled: bool):
    if is_bolza is False:
        w_opt = w_opt[1:]
    u_opt = w_opt[5::6]
    if is_scaled:
        u_opt = [x * float(params.inv_scale_u) for x in u_opt]
    return u_opt

def trajectory_computation(u_opt, k_value, s_value, A_w, B_w, x_initial=None, multiplier: int = 1, N_org: int = params.nu, integrator = rk4_step_bolza):
    N = N_org * multiplier
    dt = params.tf / N

    # State, Time and Control
    x1 = MX.sym('x1')  # x
    x2 = MX.sym('x2')  # h
    x3 = MX.sym('x3')  # V
    x4 = MX.sym('x4')  # gamma
    x5 = MX.sym('x5')  # alpha
    t = MX.sym('t')  # time
    u = MX.sym('u')  # control
    x = vertcat(x1, x2, x3, x4, x5)

    # Wind
    wind_x_expr = k_value * A_w(x1, s_value)
    wind_x = Function('wind_x', [x1], [wind_x_expr])
    wind_h_expr = k_value * x2 * B_w(x1, s_value) / params.h_star
    wind_h = Function('wind_h', [x1, x2], [wind_h_expr])

    dWx_dx = Function("dWx_dx", [x1], [gradient(wind_x_expr, x1)])
    dWh_dx = Function("dWh_dx", [x1, x2], [gradient(wind_h_expr, x1)])
    dWh_dh = Function("dWh_dh", [x1, x2], [gradient(wind_h_expr, x2)])

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2

    # ode
    x1dot = x3 * cos(x4) + wind_x(x1)
    x2dot = x3 * sin(x4) + wind_h(x1, x2)

    wxdot = dWx_dx(x1) * x1dot
    whdot = dWh_dx(x1, x2) * x1dot + dWh_dh(x1, x2) * x2dot
    x3_safe = fmax(x3, params.eps)

    x3dot = T / params.m * cos(x5 + params.delta) - D / params.m - params.g * sin(x4) - (
                wxdot * cos(x4) + whdot * sin(x4))
    x4dot = T / (params.m * x3) * sin(x5 + params.delta) + L / (params.m * x3) - params.g / x3_safe * cos(x4) + (
                1 / x3_safe) * (wxdot * sin(x4) - whdot * cos(x4))
    x5dot = u

    f = Function('f', [x, u, t], [vertcat(x1dot, x2dot, x3dot, x4dot, x5dot)])

    x2dotdot_expr = x3dot * sin(x4) + x3 * x4dot * cos(x4) + whdot
    x2dotdot = Function('x2dotdot', [x, u, t], [x2dotdot_expr])

    # objective function
    Q = (params.hR - x2) ** params.q
    qf = Function('qf', [x], [Q])

    # Prep
    U = []
    for k in range(N_org):
        for i in range(multiplier):
            U.append(u_opt[k])

    X = []
    J = 0

    # initial condition
    if x_initial is None:
        x_initial = [0, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
    Xk = x_initial
    X.extend(Xk)
    Tk = 0

    for k in range(N):
        Xk, J = integrator(f, qf, Xk, U[k], Tk, dt, J)
        Tk += dt
        X.extend(Xk.full().flatten())

    T = [params.tf / N * k for k in range(N + 1)]

    return X, U, J, T

def trajectory_computation_dictonary(u_opt: list[float], k_value, s_value, A_w, B_w, x_initial=None,multiplier: int = 1, N_org: int = params.nu, integrator = rk4_step_bolza):
    N = N_org * multiplier
    dxt = params.tf / N

    # State, Time and Control
    x1 = SX.sym('x1')  # x
    x2 = SX.sym('x2')  # h
    x3 = SX.sym('x3')  # V
    x4 = SX.sym('x4')  # gamma
    x5 = SX.sym('x5')  # alpha
    t = SX.sym('t')  # time
    u = SX.sym('u')  # control
    x = vertcat(x1, x2, x3, x4, x5)

    # Wind
    wind_x_expr = k_value * A_w(x1, s_value)
    wind_x = Function('wind_x', [x1], [wind_x_expr])
    wind_h_expr = k_value * x2 * B_w(x1, s_value) / params.h_star
    wind_h = Function('wind_h', [x1, x2], [wind_h_expr])

    dWx_dx = Function("dWx_dx", [x1], [gradient(wind_x_expr, x1)])
    dWh_dx = Function("dWh_dx", [x1, x2], [gradient(wind_h_expr, x1)])
    dWh_dh = Function("dWh_dh", [x1, x2], [gradient(wind_h_expr, x2)])

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2

    # ode
    x1dot = x3 * cos(x4) + wind_x(x1)
    x2dot = x3 * sin(x4) + wind_h(x1, x2)

    wxdot = dWx_dx(x1) * x1dot
    whdot = dWh_dx(x1, x2) * x1dot + dWh_dh(x1, x2) * x2dot
    x3_safe = fmax(x3, params.eps)

    x3dot = T / params.m * cos(x5 + params.delta) - D / params.m - params.g * sin(x4) - (
                wxdot * cos(x4) + whdot * sin(x4))
    x4dot = T / (params.m * x3) * sin(x5 + params.delta) + L / (params.m * x3) - params.g / x3_safe * cos(x4) + (
                1 / x3_safe) * (wxdot * sin(x4) - whdot * cos(x4))
    x5dot = u

    x2dotdot = x3dot * np.sin(x4) + x3 * x4dot * np.cos(x4) + whdot

    f = Function('f', [x, u, t], [vertcat(x1dot, x2dot, x3dot, x4dot, x5dot)])
    fun_hdotdot = Function('fun_hdotodt', [x, u, t], [x2dotdot])

    # objective bolza function
    Q = (params.hR - x2) ** params.q
    qf = Function('qf', [x], [Q])

    # Prep
    U = []
    for k in range(N_org):
        for i in range(multiplier):
            U.append(u_opt[k])

    X = []

    # initial condition
    if x_initial is None:
        x_initial = [0, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
    Xk = x_initial
    X.extend(Xk)
    Tk = 0

    H_dotdot = []
    J = 0

    for k in range(N):
        H_dotdot.append(fun_hdotdot(Xk, U[k], Tk).full().flatten())
        Xk,J = integrator(f,qf,Xk, U[k], Tk, dxt, J)
        Tk += dxt
        X.extend(Xk.full().flatten())

    T = [params.tf / N * k for k in range(N + 1)]

    return {'x': X[0::5],'h': X[1::5],'v': X[2::5],'gamma': X[3::5],'alpha': X[4::5],'u':U,'t_grid':T, 'hdotdot': H_dotdot, 'J': J}

def MC_simulations(u_opt, M:int, A_w, B_w, MC_type: int, k_mid, s_mid, std_k, std_s: float=0.25, x_initial=None, plot: bool=True, integrator = MC_rk4_step):
    '''
    MC_types
        1, k ~ N(k_mid, std**2)  s = s_mid
        2. k ~ N(k_mid, std**2), s = (k_mid/k)**2 (energy conserving)
        3. k ~ N(k_mid, std**2), s ~ N(s_mid, std_s)
        4.
    '''
    if MC_type not in {1, 2, 3}:
        raise ValueError("M must be 1, 2, or 3.")

    N = len(u_opt)
    dt = params.tf / N

    # State, Time and Control
    x1 = MX.sym('x1')  # x
    x2 = MX.sym('x2')  # h
    x3 = MX.sym('x3')  # V
    x4 = MX.sym('x4')  # gamma
    x5 = MX.sym('x5')  # alpha
    t = MX.sym('t')  # time
    u = MX.sym('u')  # control
    x = vertcat(x1, x2, x3, x4, x5)
    k_sym = MX.sym('k_sym')
    s_sym = MX.sym('s_sym')

    # wind
    wind_x_expr = k_sym * A_w(x1,s_sym)
    wind_x = Function('wind_x', [x1, k_sym, s_sym], [wind_x_expr])
    wind_h_expr = k_sym * x2 * B_w(x1,s_sym) / params.h_star
    wind_h = Function('wind_h', [x1, x2, k_sym, s_sym], [wind_h_expr])

    dWx_dx = Function("dWx_dx", [x1, k_sym, s_sym], [gradient(wind_x_expr, x1)])
    dWh_dx = Function("dWh_dx", [x1, x2, k_sym, s_sym], [gradient(wind_h_expr, x1)])
    dWh_dh = Function("dWh_dh", [x1, x2, k_sym, s_sym], [gradient(wind_h_expr, x2)])

    # ode
    def dynamics(x,u,t,k_value,s_value):
        x1, x2, x3, x4, x5 = x

        # Other functions
        C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                      params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

        beta = if_else(t < params.sigma,
                       params.beta0 + params.beta_dot0 * t, 1.0)

        T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

        D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

        L = 0.5 * params.rho * params.S * C_L * x3 ** 2

        x1dot = x3 * cos(x4) + wind_x(x1, k_value, s_value)
        x2dot = x3 * sin(x4) + wind_h(x1, x2, k_value, s_value)

        wxdot = dWx_dx(x1, k_value, s_value) * x1dot
        whdot = dWh_dx(x1, x2, k_value, s_value) * x1dot + dWh_dh(x1, x2, k_value, s_value) * x2dot
        x3_safe = fmax(x3, params.eps)

        x3dot = T / params.m * cos(x5 + params.delta) - D / params.m - params.g * sin(x4) - (
                wxdot * cos(x4) + whdot * sin(x4))
        x4dot = T / (params.m * x3) * sin(x5 + params.delta) + L / (params.m * x3) - params.g / x3_safe * cos(x4) + (
                1 / x3_safe) * (wxdot * sin(x4) - whdot * cos(x4))
        x5dot = u

        x2dotdot = x3dot * sin(x4) + x3 * x4dot * cos(x4) + whdot

        return np.array([
            float(x1dot.full()),
            float(x2dot.full()),
            float(x3dot.full()),
            float(x4dot.full()),
            float(x5dot)
        ]), float(x2dotdot.full())


    # prep
    k_samples = np.random.normal(loc=k_mid, scale=std_k**2, size=M)
    if MC_type == 3:
        s_samples = np.random.normal(loc=s_mid, scale=std_s**2, size=M)

    min_hs = []
    violating_ks = {
        'x2dotdot_low': [],
        'x2dotdot_high': []
    }

    if x_initial is None:
        x_initial = [0, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]

    if plot:
        plt.figure()

    for j in range(len(k_samples)):
        k_value = k_samples[j]
        xk = np.array(x_initial)
        t = 0
        traj = [xk.copy()]
        violated_low = False
        violated_high = False

        if MC_type == 1:
            s_value = s_mid
        if MC_type == 2:
            s_value = (k_mid/k_value)**2
        if MC_type == 3:
            s_value = s_samples[j]

        for i in range(params.nu):
            xk, x2dotdot = integrator(dynamics, xk, u_opt[i], t, dt, k_value, s_value)
            traj.append(xk.copy())
            t += params.dut

            if x2dotdot < -2 * params.g:
                violated_low = True
            if x2dotdot > 10 * params.g:
                violated_high = True

        if violated_low:
            violating_ks['x2dotdot_low'].append(float(k_value))
        if violated_high:
            violating_ks['x2dotdot_high'].append(float(k_value))

        traj = np.array(traj)
        h_list = traj[:, 1]
        min_h = min(h_list)
        min_hs.append(min_h)

        if plot:
            color = 'red' if min_h <= 0 else 'orange' if min_h <= 50 else 'gold' if min_h <= 100 else 'green'
            plt.plot(traj[:, 0], traj[:, 1], color=color)

    if plot:
        plt.xlabel('Horizontal Distance')
        plt.ylabel('Altitude [ft]')
        plt.grid(True)
        plt.show()

    print('min k from sample: ' + str(f"{min(k_samples):.3f}"))
    print('max k from sample: ' + str(f"{max(k_samples):.3f}"))
    if MC_type == 2:
        print('min s from sample: ' + str(f"{(k_mid/max(k_samples))**2:.3f}"))
        print('max s from sample: ' + str(f"{(k_mid/min(k_samples))**2:.3f}"))
    if MC_type == 3:
        print('min s from sample: ' + str(f"{min(s_samples):.3f}"))
        print('max s from sample: ' + str(f"{max(s_samples):.3f}"))
    print('Minimum altitude over all runs: ' + str(f"{min(min_hs):.3f}"))

    M = len(min_hs)
    crash_0 = math.fsum(h <= 0 for h in min_hs) / M * 100
    fail_50 = math.fsum(h <= 50 for h in min_hs) / M * 100
    fail_100 = math.fsum(h <= 100 for h in min_hs) / M * 100

    print('Crash rate: ' + str(crash_0) + '%')
    print('Fail 50 feet rate: ' + str(fail_50) + '%')
    print('Fail 100 rate: ' + str(fail_100) + '%')

    return violating_ks

def sol_dictonary(w_opt: list[float], is_bolza: bool, is_scaled: bool):
    if is_bolza is False:
        w_opt = w_opt[1:]
    x1_opt = w_opt[0::6]  # start at index 0, get every 6th:
    x2_opt = w_opt[1::6]
    x3_opt = w_opt[2::6]
    x4_opt = w_opt[3::6]
    x5_opt = w_opt[4::6]
    u_opt = w_opt[5::6]

    if is_scaled:
        x1_opt = [x * float(params.inv_scale_x[0]) for x in x1_opt]
        x2_opt = [x * float(params.inv_scale_x[1]) for x in x2_opt]
        x3_opt = [x * float(params.inv_scale_x[2]) for x in x3_opt]
        x4_opt = [x * float(params.inv_scale_x[3]) for x in x4_opt]
        x5_opt = [x * float(params.inv_scale_x[4]) for x in x5_opt]
        u_opt = [x * float(params.inv_scale_u) for x in u_opt]

    N = len(u_opt)
    tf = params.tf
    t_grid = [tf / N * k for k in range(N + 1)]
    return {'x': x1_opt,'h': x2_opt,'v': x3_opt,'gamma': x4_opt,'alpha': x5_opt,'u': u_opt,'t_grid': t_grid}













# ---------- Below not working or not important -------------

'''
def solver_min_h_k_sym(k_value, N: int = params.nu):
    params.nu = N
    params.dut = params.tf / params.nu

    # State, Time and Control
    x1 = MX.sym('x1')  # x
    x2 = MX.sym('x2')  # h
    x3 = MX.sym('x3')  # V
    x4 = MX.sym('x4')  # gamma
    x5 = MX.sym('x5')  # alpha
    t = MX.sym('t')  # time
    u = MX.sym('u')  # control
    k_sym = MX.sym('k_sym')  # wind parameter
    x = vertcat(x1, x2, x3, x4, x5)

    # Wind
    A1 = -50 + params.a * x1 ** 3 + params.b * x1 ** 4
    A2 = 0.025 * (x1 - 2300)
    A3 = 50 - params.a * (4600 - x1) ** 3 - params.b * (4600 - x1) ** 4
    A4 = 50
    A = if_else(x1 <= 500, A1,
                if_else(x1 <= 4100, A2,
                        if_else(x1 <= 4600, A3, A4)))

    B1 = params.d * x1 ** 3 + params.e * x1 ** 4
    B2 = -51 * exp(-params.c * (x1 - 2300) ** 4)
    B3 = params.d * (4600 - x1) ** 3 + params.e * (4600 - x1) ** 4
    B4 = 0
    B = if_else(x1 <= 500, B1,
                if_else(x1 <= 4100, B2,
                        if_else(x1 <= 4600, B3, B4)))

    wind_x_expr = k_sym * A
    wind_x = Function('wind_x', [x1, k_sym], [wind_x_expr])
    wind_h_expr = k_sym * x2 * B / params.h_star
    wind_h = Function('wind_h', [x1, x2, k_sym], [wind_h_expr])

    dWx_dx = Function("dWx_dx", [x1, k_sym], [gradient(wind_x_expr, x1)])
    dWh_dx = Function("dWh_dx", [x1, x2, k_sym], [gradient(wind_h_expr, x1)])
    dWh_dh = Function("dWh_dh", [x1, x2, k_sym], [gradient(wind_h_expr, x2)])

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2

    # ode
    x1dot = x3 * cos(x4) + wind_x(x1, k_sym)
    x2dot = x3 * sin(x4) + wind_h(x1, x2, k_sym)

    wxdot = dWx_dx(x1, k_sym) * x1dot
    whdot = dWh_dx(x1, x2, k_sym) * x1dot + dWh_dh(x1, x2, k_sym) * x2dot
    x3_safe = fmax(x3, params.eps)

    x3dot = T / params.m * cos(x5 + params.delta) - D / params.m - params.g * sin(x4) - (
            wxdot * cos(x4) + whdot * sin(x4))
    x4dot = T / (params.m * x3) * sin(x5 + params.delta) + L / (params.m * x3) - params.g / x3_safe * cos(x4) + (
            1 / x3_safe) * (wxdot * sin(x4) - whdot * cos(x4))
    x5dot = u

    f = Function('f', [x, u, t, k_sym], [vertcat(x1dot, x2dot, x3dot, x4dot, x5dot)])

    x2dotdot_expr = x3dot * sin(x4) + x3 * x4dot * cos(x4) + whdot
    x2dotdot = Function('x2dotdot', [x, u, t, k_sym], [x2dotdot_expr])

    def rk4_step(xk, uk, tk, dt, k_val):
        k1 = f(xk, uk, tk, k_val)
        k2 = f(xk + dt / 2 * k1, uk, tk + dt / 2, k_val)
        k3 = f(xk + dt / 2 * k2, uk, tk + dt / 2, k_val)
        k4 = f(xk + dt * k3, uk, tk + dt, k_val)
        return xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    w = []
    w0 = []
    lbw = []
    ubw = []
    g = []
    lbg = []
    ubg = []
    J = 0

    min_h = MX.sym('min_h')
    w += [min_h]
    lbw += [0]
    ubw += [inf]
    w0 += [600]

    x_initial = [0, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
    Xk = MX.sym('X_0', 5)
    w += [Xk]
    lbw += x_initial
    ubw += x_initial
    w0 += x_initial

    Tk = 0

    for k in range(params.nu):
        Uk = MX.sym('U_' + str(k))
        w += [Uk]
        lbw += [-3 * pi / 180]
        ubw += [3 * pi / 180]
        w0 += [0]

        Xk_end = rk4_step(Xk, Uk, Tk, params.dut, k_sym)
        h_ddot_val = x2dotdot(Xk, Uk, Tk, k_sym)

        Tk += params.dut

        Xk = MX.sym('X_' + str(k + 1), 5)
        w += [Xk]
        lbw += [-inf, 0, -inf, -inf, -inf]
        ubw += [inf, inf, inf, inf, 17.2 * pi / 180]
        w0 += [k * 10000 / params.nu, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
        g += [Xk_end - Xk]
        lbg += [0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0]

        g += [Xk[1] - min_h]
        lbg += [0]
        ubg += [inf]

        g += [h_ddot_val]
        lbg += [-2 * params.g]
        ubg += [10 * params.g]

    XF = 7.431 * pi / 180
    g += [Xk[3] - XF]
    lbg += [0]
    ubg += [0]

    J = -min_h

    opts = {'ipopt': {'print_level': 3}}
    npl = {'x': vertcat(*w), 'f': J, 'g': vertcat(*g), 'p': k_sym}
    solver = nlpsol('solver', 'ipopt', npl, opts)

    arg = {'x0': w0, 'lbx': lbw, 'ubx': ubw, 'lbg': lbg, 'ubg': ubg, 'p': k_value}

    sol = solver(**arg)
    w_opt = sol['x'].full().flatten()
    J_opt = sol['f'].full().flatten()

    return w_opt, J_opt

def solver_min_h_scaled_k_sym(k_value, N: int = params.nu):
    params.nu = N
    params.dut = params.tf / params.nu

    # State, Time and Control
    x1 = MX.sym('x1')  # x
    x2 = MX.sym('x2')  # h
    x3 = MX.sym('x3')  # V
    x4 = MX.sym('x4')  # gamma
    x5 = MX.sym('x5')  # alpha
    t = MX.sym('t')  # time
    u = MX.sym('u')  # control
    k_sym = MX.sym('k_sym')  # wind parameter
    x = vertcat(x1, x2, x3, x4, x5)

    # Wind
    A1 = -50 + params.a * x1 ** 3 + params.b * x1 ** 4
    A2 = 0.025 * (x1 - 2300)
    A3 = 50 - params.a * (4600 - x1) ** 3 - params.b * (4600 - x1) ** 4
    A4 = 50
    A = if_else(x1 <= 500, A1,
                if_else(x1 <= 4100, A2,
                        if_else(x1 <= 4600, A3, A4)))

    B1 = params.d * x1 ** 3 + params.e * x1 ** 4
    B2 = -51 * exp(-params.c * (x1 - 2300) ** 4)
    B3 = params.d * (4600 - x1) ** 3 + params.e * (4600 - x1) ** 4
    B4 = 0
    B = if_else(x1 <= 500, B1,
                if_else(x1 <= 4100, B2,
                        if_else(x1 <= 4600, B3, B4)))

    wind_x_expr = k_sym * A
    wind_x = Function('wind_x', [x,k_sym], [wind_x_expr])
    wind_h_expr = k_sym * x2 * B / params.h_star
    wind_h = Function('wind_h', [x,k_sym], [wind_h_expr])

    dWx_dx = Function("dWx_dx", [x,k_sym], [gradient(wind_x_expr, x1)])
    dWh_dx = Function("dWh_dx", [x,k_sym], [gradient(wind_h_expr, x1)])
    dWh_dh = Function("dWh_dh", [x,k_sym], [gradient(wind_h_expr, x2)])

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2

    # ode
    x1dot = x3 * cos(x4) + wind_x(x,k_sym)
    x2dot = x3 * sin(x4) + wind_h(x,k_sym)

    wxdot = dWx_dx(x,k_sym) * x1dot
    whdot = dWh_dx(x,k_sym) * x1dot + dWh_dh(x,k_sym) * x2dot
    x3_safe = fmax(x3, params.eps)

    x3dot = T / params.m * cos(x5 + params.delta) - D / params.m - params.g * sin(x4) - (
            wxdot * cos(x4) + whdot * sin(x4))
    x4dot = T / (params.m * x3) * sin(x5 + params.delta) + L / (params.m * x3) - params.g / x3_safe * cos(x4) + (
            1 / x3_safe) * (wxdot * sin(x4) - whdot * cos(x4))
    x5dot = u

    f_org = Function('f_org', [x, u, t, k_sym], [vertcat(x1dot, x2dot, x3dot, x4dot, x5dot)])

    x2dotdot_expr = x3dot * sin(x4) + x3 * x4dot * cos(x4) + whdot
    x2dotdot = Function('x2dotdot', [x, u, t, k_sym], [x2dotdot_expr])

    # scaled state, control and sys dynamics
    xs = MX.sym('xs', 5)
    us = MX.sym('us')
    f_s = params.scale_x * f_org(params.inv_scale_x * xs, params.inv_scale_u * us, t, k_sym)
    f_scaled = Function('f_scaled', [xs, us, t, k_sym], [f_s])

    # Integration
    def rk4_step(xsk, usk, tk, dt, k_val):
        k1 = f_scaled(xsk, usk, tk, k_val)
        k2 = f_scaled(xsk + dt / 2 * k1, usk, tk + dt / 2, k_val)
        k3 = f_scaled(xsk + dt / 2 * k2, usk, tk + dt / 2, k_val)
        k4 = f_scaled(xsk + dt * k3, usk, tk + dt, k_val)
        return xsk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # npl prep
    w = []  # List of decision variables
    w0 = []  # Initial guess
    lbw = []  # Lower bounds
    ubw = []  # Upper bounds
    g = []  # Constraints
    lbg = []  # Lower bound on constraints
    ubg = []  # Upper bound on constraints
    J = 0  # Objective function

    # initial condition
    min_h_s = MX.sym('min_h_s')
    w += [min_h_s]
    lbw += [0]
    ubw += [inf]
    w0 += [600*float(params.scale_h)]

    x_initial = [0, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
    xs_initial = [x_initial[i] * float(params.scale_x[i]) for i in range(5)]
    Xsk = MX.sym('Xs_0', 5)
    w += [Xsk]
    lbw += xs_initial
    ubw += xs_initial
    w0 += xs_initial

    Tk = 0

    # npl set up
    for k in range(params.nu):
        # control
        Usk = MX.sym('Us_' + str(k))
        w += [Usk]
        lbw += [-3 * pi / 180 * params.scale_u]
        ubw += [3 * pi / 180 * params.scale_u]
        w0 += [0]

        # state integration
        Xsk_end = rk4_step(Xsk, Usk, Tk, params.dut, k_sym)
        h_ddot_val = x2dotdot(Xsk*params.inv_scale_x, Usk*params.inv_scale_u, Tk, k_sym)

        Tk += params.dut

        # state
        Xsk = MX.sym('Xs_' + str(k + 1), 5)
        w += [Xsk]
        lbw += [-inf, 0, -inf, -inf, -inf]
        ubw += [inf, inf, inf, inf, 17.2 * pi / 180 * params.scale_x[4]]
        w0 += [k * 10000 / params.nu * params.scale_x[0], 600 * params.scale_x[1], 239.7* params.scale_x[2], -2.249 * pi / 180 * params.scale_x[3], 7.353 * pi / 180 * params.scale_x[4]]
        g += [Xsk_end - Xsk]
        lbg += [0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0]

        g += [Xsk[1] - min_h_s]
        lbg += [0]
        ubg += [inf]

        # Add to constraint list
        g += [h_ddot_val]
        lbg += [-2 * params.g]
        ubg += [10 * params.g]

    XF = 7.431 * pi / 180 * float(params.scale_x[3])
    g += [Xsk[3] - XF]
    lbg += [0]
    ubg += [0]

    J = -min_h_s*params.inv_scale_h

    opts = {'ipopt': { 'print_level': 3}}
    npl = {'x': vertcat(*w), 'f': J, 'g': vertcat(*g), 'p': k_sym}
    solver = nlpsol('solver', 'ipopt', npl, opts)

    arg = {'x0': vertcat(*w0), 'lbx': vertcat(*lbw), 'ubx': vertcat(*ubw), 'lbg': vertcat(*lbg), 'ubg': vertcat(*ubg), 'p':k_value}

    # Solve
    sol = solver(**arg)
    w_opt = sol['x'].full().flatten()
    J_opt = sol['f'].full().flatten()

    return w_opt, J_opt

def solver_bolza_smooth(k_value, N: int = params.nu):
    params.nu = N
    params.dut = params.tf / params.nu

    # State, Time and Control
    x1 = MX.sym('x1')  # x
    x2 = MX.sym('x2')  # h
    x3 = MX.sym('x3')  # V
    x4 = MX.sym('x4')  # gamma
    x5 = MX.sym('x5')  # alpha
    t = MX.sym('t')  # time
    u = MX.sym('u')  # control
    x = vertcat(x1, x2, x3, x4, x5)

    # Wind
    def Smooth(x_start, x_end):
        t_smooth = (x1 - x_start) / (x_end - x_start + params.eps)
        return if_else(x1 < x_start, 0,
                       if_else(x1 > x_end, 1, 6 * t_smooth ** 5 - 15 * t_smooth ** 4 + 10 * t_smooth ** 3))

    A1 = -50 + params.a * x1 ** 3 + params.b * x1 ** 4
    A2 = 0.025 * (x1 - 2300)
    A3 = 50 - params.a * (4600 - x1) ** 3 - params.b * (4600 - x1) ** 4
    A4 = 50
    s1 = Smooth(480, 520)
    s2 = Smooth(4080, 4120)
    s3 = Smooth(4580, 4620)
    A = if_else(x1 <= 500, (1 - s1) * A1 + s1 * A2,
                if_else(x1 <= 4100, (1 - s2) * A2 + s2 * A3,
                        if_else(x1 <= 4600, (1 - s3) * A3 + s3 * A4, A4)))

    B1 = params.d * x1 ** 3 + params.e * x1 ** 4
    B2 = -51 * exp(-params.c * (x1 - 2300) ** 4)
    B3 = params.d * (4600 - x1) ** 3 + params.e * (4600 - x1) ** 4
    B4 = 0
    s1 = Smooth(480, 520)
    s2 = Smooth(4080, 4120)
    s3 = Smooth(4580, 4620)
    B = if_else(x1 <= 500, (1 - s1) * B1 + s1 * B2,
                if_else(x1 <= 4100, (1 - s2) * B2 + s2 * B3,
                        if_else(x1 <= 4600, (1 - s3) * B3 + s3 * B4, B4)))

    wind_x_expr = k_value * A
    wind_x = Function('wind_x', [x1], [wind_x_expr])
    wind_h_expr = k_value * x2 * B / params.h_star
    wind_h = Function('wind_h', [x1, x2], [wind_h_expr])

    dWx_dx = Function("dWx_dx", [x1], [gradient(wind_x_expr, x1)])
    dWh_dx = Function("dWh_dx", [x1, x2], [gradient(wind_h_expr, x1)])
    dWh_dh = Function("dWh_dh", [x1, x2], [gradient(wind_h_expr, x2)])

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2

    # ode
    x1dot = x3 * cos(x4) + wind_x(x1)
    x2dot = x3 * sin(x4) + wind_h(x1, x2)

    wxdot = dWx_dx(x1) * x1dot
    whdot = dWh_dx(x1, x2) * x1dot + dWh_dh(x1, x2) * x2dot
    x3_safe = fmax(x3, params.eps)

    x3dot = T / params.m * cos(x5 + params.delta) - D / params.m - params.g * sin(x4) - (
                wxdot * cos(x4) + whdot * sin(x4))
    x4dot = T / (params.m * x3) * sin(x5 + params.delta) + L / (params.m * x3) - params.g / x3_safe * cos(x4) + (
                1 / x3_safe) * (wxdot * sin(x4) - whdot * cos(x4))
    x5dot = u

    f = Function('f', [x, u, t], [vertcat(x1dot, x2dot, x3dot, x4dot, x5dot)])

    x2dotdot_expr = x3dot * sin(x4) + x3 * x4dot * cos(x4) + whdot
    x2dotdot = Function('x2dotdot', [x, u, t], [x2dotdot_expr])

    # objective function
    Q = (params.hR - x2) ** params.q
    qf = Function('qf', [x], [Q])

    # Integration
    def rk4_step(xk, uk, tk, dt, qk):
        k1 = f(xk, uk, tk)
        k2 = f(xk + dt / 2 * k1, uk, tk + dt / 2)
        k3 = f(xk + dt / 2 * k2, uk, tk + dt / 2)
        k4 = f(xk + dt * k3, uk, tk + dt)
        qk1 = qf(xk)
        qk2 = qf(xk + dt / 2 * k1)
        qk3 = qf(xk + dt / 2 * k2)
        qk4 = qf(xk + dt * k3)
        return xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), qk + dt / 6 * (qk1 + 2 * qk2 + 2 * qk3 + qk4)

    # npl prep
    w = []  # List of decision variables
    w0 = []  # Initial guess
    lbw = []  # Lower bounds
    ubw = []  # Upper bounds
    g = []  # Constraints
    lbg = []  # Lower bound on constraints
    ubg = []  # Upper bound on constraints
    J = 0  # Objective function

    # initial condition
    x_initial = [0, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
    Xk = MX.sym('X_0', 5)
    w += [Xk]
    lbw += x_initial
    ubw += x_initial
    w0 += x_initial

    Tk = 0

    # npl set up
    for k in range(params.nu):
        # control
        Uk = MX.sym('U_' + str(k))
        w += [Uk]
        lbw += [-3 * pi / 180]
        ubw += [3 * pi / 180]
        w0 += [0]

        # state integration
        Xk_end, J_end = rk4_step(Xk, Uk, Tk, params.dut, J)
        J = J_end
        h_ddot_val = x2dotdot(Xk, Uk, Tk)

        Tk += params.dut

        # state
        Xk = MX.sym('X_' + str(k + 1), 5)
        w += [Xk]
        lbw += [-inf, 0, -inf, -inf, -inf]
        ubw += [inf, params.hR, inf, inf, 17.2 * pi / 180]
        w0 += [k * 10000 / params.nu, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
        g += [Xk_end - Xk]
        lbg += [0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0]

        # Add to constraint list
        g += [h_ddot_val]
        lbg += [-2 * params.g]
        ubg += [10 * params.g]

    XF = 7.431 * pi / 180
    g += [Xk[3] - XF]
    lbg += [0]
    ubg += [0]

    opts = {'ipopt': {'print_level': 3}}

    npl = {'x': vertcat(*w), 'f': J, 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', npl, opts)

    arg = {'x0': w0, 'lbx': lbw, 'ubx': ubw, 'lbg': lbg, 'ubg': ubg}

    # Solve
    sol = solver(**arg)
    w_opt = sol['x'].full().flatten()
    J_opt = sol['f'].full().flatten()

    return w_opt, J_opt

def solver_min_h_smooth(k_value: float, N: int = params.nu):
    params.nu = N
    params.dut = params.tf / params.nu

    # State, Time and Control
    x1 = MX.sym('x1')  # x
    x2 = MX.sym('x2')  # h
    x3 = MX.sym('x3')  # V
    x4 = MX.sym('x4')  # gamma
    x5 = MX.sym('x5')  # alpha
    t = MX.sym('t')  # time
    u = MX.sym('u')  # control
    x = vertcat(x1, x2, x3, x4, x5)

    # Wind
    def Smooth(x_start, x_end):
        t_smooth = (x1 - x_start) / (x_end - x_start + params.eps)
        return if_else(x1 < x_start, 0,
                       if_else(x1 > x_end, 1, 6 * t_smooth ** 5 - 15 * t_smooth ** 4 + 10 * t_smooth ** 3))

    A1 = -50 + params.a * x1 ** 3 + params.b * x1 ** 4
    A2 = 0.025 * (x1 - 2300)
    A3 = 50 - params.a * (4600 - x1) ** 3 - params.b * (4600 - x1) ** 4
    A4 = 50
    s1 = Smooth(480, 520)
    s2 = Smooth(4080, 4120)
    s3 = Smooth(4580, 4620)
    A = if_else(x1 <= 500, (1 - s1) * A1 + s1 * A2,
                if_else(x1 <= 4100, (1 - s2) * A2 + s2 * A3,
                        if_else(x1 <= 4600, (1 - s3) * A3 + s3 * A4, A4)))

    B1 = params.d * x1 ** 3 + params.e * x1 ** 4
    B2 = -51 * exp(-params.c * (x1 - 2300) ** 4)
    B3 = params.d * (4600 - x1) ** 3 + params.e * (4600 - x1) ** 4
    B4 = 0
    s1 = Smooth(480, 520)
    s2 = Smooth(4080, 4120)
    s3 = Smooth(4580, 4620)
    B = if_else(x1 <= 500, (1 - s1) * B1 + s1 * B2,
                if_else(x1 <= 4100, (1 - s2) * B2 + s2 * B3,
                        if_else(x1 <= 4600, (1 - s3) * B3 + s3 * B4, B4)))

    wind_x_expr = k_value * A
    wind_x = Function('wind_x', [x1], [wind_x_expr])
    wind_h_expr = k_value * x2 * B / params.h_star
    wind_h = Function('wind_h', [x1, x2], [wind_h_expr])

    dWx_dx = Function("dWx_dx", [x1], [gradient(wind_x_expr, x1)])
    dWh_dx = Function("dWh_dx", [x1, x2], [gradient(wind_h_expr, x1)])
    dWh_dh = Function("dWh_dh", [x1, x2], [gradient(wind_h_expr, x2)])

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2

    # ode
    x1dot = x3 * cos(x4) + wind_x(x1)
    x2dot = x3 * sin(x4) + wind_h(x1, x2)

    wxdot = dWx_dx(x1) * x1dot
    whdot = dWh_dx(x1, x2) * x1dot + dWh_dh(x1, x2) * x2dot
    x3_safe = fmax(x3, params.eps)

    x3dot = T / params.m * cos(x5 + params.delta) - D / params.m - params.g * sin(x4) - (
            wxdot * cos(x4) + whdot * sin(x4))
    x4dot = T / (params.m * x3) * sin(x5 + params.delta) + L / (params.m * x3) - params.g / x3_safe * cos(x4) + (
            1 / x3_safe) * (wxdot * sin(x4) - whdot * cos(x4))
    x5dot = u

    f = Function('f', [x, u, t], [vertcat(x1dot, x2dot, x3dot, x4dot, x5dot)])

    x2dotdot_expr = x3dot * sin(x4) + x3 * x4dot * cos(x4) + whdot
    x2dotdot = Function('x2dotdot', [x, u, t], [x2dotdot_expr])

    # Integration
    def rk4_step(xk, uk, tk, dt):
        k1 = f(xk, uk, tk)
        k2 = f(xk + dt / 2 * k1, uk, tk + dt / 2)
        k3 = f(xk + dt / 2 * k2, uk, tk + dt / 2)
        k4 = f(xk + dt * k3, uk, tk + dt)
        return xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # npl prep
    w = []  # List of decision variables
    w0 = []  # Initial guess
    lbw = []  # Lower bounds
    ubw = []  # Upper bounds
    g = []  # Constraints
    lbg = []  # Lower bound on constraints
    ubg = []  # Upper bound on constraints
    J = 0  # Objective function

    # initial condition
    min_h = MX.sym('min_h')
    w += [min_h]
    lbw += [0]
    ubw += [inf]
    w0 += [600]

    x_initial = [0, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
    Xk = MX.sym('X_0', 5)
    w += [Xk]
    lbw += x_initial
    ubw += x_initial
    w0 += x_initial

    Tk = 0

    # npl set up
    for k in range(params.nu):
        # control
        Uk = MX.sym('U_' + str(k))
        w += [Uk]
        lbw += [-3 * pi / 180]
        ubw += [3 * pi / 180]
        w0 += [0]

        # state integration
        Xk_end = rk4_step(Xk, Uk, Tk, params.dut)
        h_ddot_val = x2dotdot(Xk, Uk, Tk)

        Tk += params.dut

        # state
        Xk = MX.sym('X_' + str(k + 1), 5)
        w += [Xk]
        lbw += [-inf, 0, -inf, -inf, -inf]
        ubw += [inf, inf, inf, inf, 17.2 * pi / 180]
        w0 += [k * 10000 / params.nu, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
        g += [Xk_end - Xk]
        lbg += [0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0]

        g += [Xk[1] - min_h]
        lbg += [0]
        ubg += [inf]

        # Add to constraint list
        g += [h_ddot_val]
        lbg += [-2 * params.g]
        ubg += [10 * params.g]

    XF = 7.431 * pi / 180
    g += [Xk[3] - XF]
    lbg += [0]
    ubg += [0]

    J = -min_h

    opts = {'ipopt': { 'print_level': 3}}
    npl = {'x': vertcat(*w), 'f': J, 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', npl, opts)

    arg = {'x0': w0, 'lbx': lbw, 'ubx': ubw, 'lbg': lbg, 'ubg': ubg}

    # Solve
    sol = solver(**arg)
    w_opt = sol['x'].full().flatten()
    J_opt = sol['f'].full().flatten()
    return w_opt, J_opt

def solver_min_h_sa_k_sym(k_value, N: int = params.nu):
    params.nu = N
    params.dut = params.tf / params.nu

    # State, Time and Control
    x1 = MX.sym('x1')  # x
    x2 = MX.sym('x2')  # h
    x3 = MX.sym('x3')  # V
    x4 = MX.sym('x4')  # gamma
    x5 = MX.sym('x5')  # alpha
    t = MX.sym('t')  # time
    u = MX.sym('u')  # control
    k_sym = MX.sym('k_sym')  # wind parameter
    x = vertcat(x1, x2, x3, x4, x5)

    # Wind
    A1 = -50 + params.a * x1 ** 3 + params.b * x1 ** 4
    A2 = 0.025 * (x1 - 2300)
    A3 = 50 - params.a * (4600 - x1) ** 3 - params.b * (4600 - x1) ** 4
    A4 = 50
    A = if_else(x1 <= 500, A1,
                if_else(x1 <= 4100, A2,
                        if_else(x1 <= 4600, A3, A4)))

    B1 = params.d * x1 ** 3 + params.e * x1 ** 4
    B2 = -51 * exp(-params.c * (x1 - 2300) ** 4)
    B3 = params.d * (4600 - x1) ** 3 + params.e * (4600 - x1) ** 4
    B4 = 0
    B = if_else(x1 <= 500, B1,
                if_else(x1 <= 4100, B2,
                        if_else(x1 <= 4600, B3, B4)))

    wind_x_expr = k_sym * A
    wind_x = Function('wind_x', [x1, k_sym], [wind_x_expr])
    wind_h_expr = k_sym * x2 * B / params.h_star
    wind_h = Function('wind_h', [x1, x2, k_sym], [wind_h_expr])

    dWx_dx = Function("dWx_dx", [x1, k_sym], [gradient(wind_x_expr, x1)])
    dWh_dx = Function("dWh_dx", [x1, x2, k_sym], [gradient(wind_h_expr, x1)])
    dWh_dh = Function("dWh_dh", [x1, x2, k_sym], [gradient(wind_h_expr, x2)])

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2

    # ode
    x1dot = x3 * cos(x4) + wind_x(x1, k_sym)
    x2dot = x3 * sin(x4) + wind_h(x1, x2, k_sym)

    wxdot = dWx_dx(x1, k_sym) * x1dot
    whdot = dWh_dx(x1, x2, k_sym) * x1dot + dWh_dh(x1, x2, k_sym) * x2dot
    x3_safe = fmax(x3, params.eps)

    x3dot = T / params.m * cos(x5 + params.delta) - D / params.m - params.g * sin(x4) - (
            wxdot * cos(x4) + whdot * sin(x4))
    x4dot = T / (params.m * x3) * sin(x5 + params.delta) + L / (params.m * x3) - params.g / x3_safe * cos(x4) + (
            1 / x3_safe) * (wxdot * sin(x4) - whdot * cos(x4))
    x5dot = u

    f = Function('f', [x, u, t, k_sym], [vertcat(x1dot, x2dot, x3dot, x4dot, x5dot)])

    x2dotdot_expr = x3dot * sin(x4) + x3 * x4dot * cos(x4) + whdot
    x2dotdot = Function('x2dotdot', [x, u, t, k_sym], [x2dotdot_expr])

    def rk4_step(xk, uk, tk, dt, k_val):
        k1 = f(xk, uk, tk, k_val)
        k2 = f(xk + dt / 2 * k1, uk, tk + dt / 2, k_val)
        k3 = f(xk + dt / 2 * k2, uk, tk + dt / 2, k_val)
        k4 = f(xk + dt * k3, uk, tk + dt, k_val)
        return xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    w = []
    w0 = []
    lbw = []
    ubw = []
    g = []
    lbg = []
    ubg = []
    J = 0

    min_h = MX.sym('min_h')
    w += [min_h]
    lbw += [0]
    ubw += [inf]
    w0 += [600]

    x_initial = [0, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
    Xk = MX.sym('X_0', 5)
    w += [Xk]
    lbw += x_initial
    ubw += x_initial
    w0 += x_initial

    Tk = 0

    for k in range(params.nu):
        Uk = MX.sym('U_' + str(k))
        w += [Uk]
        lbw += [-3 * pi / 180]
        ubw += [3 * pi / 180]
        w0 += [0]

        Xk_end = rk4_step(Xk, Uk, Tk, params.dut, k_sym)
        h_ddot_val = x2dotdot(Xk, Uk, Tk, k_sym)

        Tk += params.dut

        Xk = MX.sym('X_' + str(k + 1), 5)
        w += [Xk]
        lbw += [-inf, 0, -inf, -inf, -inf]
        ubw += [inf, inf, inf, inf, 17.2 * pi / 180]
        w0 += [k * 10000 / params.nu, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
        g += [Xk_end - Xk]
        lbg += [0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0]

        g += [Xk[1] - min_h]
        lbg += [0]
        ubg += [inf]

        g += [h_ddot_val]
        lbg += [-2 * params.g]
        ubg += [10 * params.g]

    XF = 7.431 * pi / 180
    g += [Xk[3] - XF]
    lbg += [0]
    ubg += [0]

    J = -min_h

    opts = {'ipopt': {'print_level': 3}}
    npl = {'x': vertcat(*w), 'f': J, 'g': vertcat(*g), 'p': k_sym}
    solver = nlpsol('solver', 'ipopt', npl, opts)

    arg = {'x0': w0, 'lbx': lbw, 'ubx': ubw, 'lbg': lbg, 'ubg': ubg, 'p': k_value}

    sol = solver(**arg)
    w_opt = sol['x'].full().flatten()
    J_opt = sol['f'].full().flatten()
    lam_g_opt = sol['lam_g'].full().flatten()
    lam_w_opt = sol['lam_x'].full().flatten()

    # ========== Sensitivity Analysis ==========
    g = vertcat(*g)
    w = vertcat(*w)
    # KKT variables: w and λ
    lam_g = MX.sym("lam", g.shape[0])
    lam_w = MX.sym("lam", w.shape[0])
    L = J + dot(lam_g, g) + dot(lam_w, w)  # Lagrangian

    # KKT conditions
    grad_L = gradient(L, w)
    comp_slack_g = diag(lam_g) @ g
    comp_slack_w = diag(lam_w) @ w

    # F(z,p)
    F = vertcat(grad_L, comp_slack_g, comp_slack_w)
    z = vertcat(w, lam_g, lam_w)

    # Compute Jacobians
    Fz = jacobian(F, z)
    Fp = jacobian(F,k_sym)

    # Evaluate at solution
    z_val = np.concatenate([w_opt, lam_g_opt, lam_w_opt])  # z^*

    Fz_func = Function("Fz", [z, k_sym], [Fz])
    Fp_func = Function("Fp", [z, k_sym], [Fp])

    Fz_eval = Fz_func(z_val, k_value)
    Fp_eval = Fp_func(z_val, k_value)

    # Solve for dz/dp
    dzdp = -np.linalg.solve(Fz_eval, Fp_eval)
    #dzdp = -np.linalg.lstsq(Fz_eval, Fp_eval, rcond=None)[0]
    dwdp = dzdp[0:w.shape[0]]
    print(np.linalg.cond(Fz_eval))

    return w_opt, J_opt, dwdp

def solver_min_h_sa_xi_sym(k_value, x_initial: list[float] = [0, 600, 239.7, -2.249 * np.pi / 180, 7.353 * np.pi / 180], N: int = params.nu):
    params.nu = N
    params.dut = params.tf / params.nu

    # State, Time and Control
    x1 = MX.sym('x1')  # x
    x2 = MX.sym('x2')  # h
    x3 = MX.sym('x3')  # V
    x4 = MX.sym('x4')  # gamma
    x5 = MX.sym('x5')  # alpha
    t = MX.sym('t')  # time
    u = MX.sym('u')  # control
    x = vertcat(x1, x2, x3, x4, x5)

    # Wind
    A1 = -50 + params.a * x1 ** 3 + params.b * x1 ** 4
    A2 = 0.025 * (x1 - 2300)
    A3 = 50 - params.a * (4600 - x1) ** 3 - params.b * (4600 - x1) ** 4
    A4 = 50
    A = if_else(x1 <= 500, A1,
                if_else(x1 <= 4100, A2,
                        if_else(x1 <= 4600, A3, A4)))

    B1 = params.d * x1 ** 3 + params.e * x1 ** 4
    B2 = -51 * exp(-params.c * (x1 - 2300) ** 4)
    B3 = params.d * (4600 - x1) ** 3 + params.e * (4600 - x1) ** 4
    B4 = 0
    B = if_else(x1 <= 500, B1,
                if_else(x1 <= 4100, B2,
                        if_else(x1 <= 4600, B3, B4)))

    wind_x_expr = k_value * A
    wind_x = Function('wind_x', [x1], [wind_x_expr])
    wind_h_expr = k_value * x2 * B / params.h_star
    wind_h = Function('wind_h', [x1, x2], [wind_h_expr])

    dWx_dx = Function("dWx_dx", [x1], [gradient(wind_x_expr, x1)])
    dWh_dx = Function("dWh_dx", [x1, x2], [gradient(wind_h_expr, x1)])
    dWh_dh = Function("dWh_dh", [x1, x2], [gradient(wind_h_expr, x2)])

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2

    # ode
    x1dot = x3 * cos(x4) + wind_x(x1)
    x2dot = x3 * sin(x4) + wind_h(x1, x2)

    wxdot = dWx_dx(x1) * x1dot
    whdot = dWh_dx(x1, x2) * x1dot + dWh_dh(x1, x2) * x2dot
    x3_safe = fmax(x3, params.eps)

    x3dot = T / params.m * cos(x5 + params.delta) - D / params.m - params.g * sin(x4) - (
            wxdot * cos(x4) + whdot * sin(x4))
    x4dot = T / (params.m * x3) * sin(x5 + params.delta) + L / (params.m * x3) - params.g / x3_safe * cos(x4) + (
            1 / x3_safe) * (wxdot * sin(x4) - whdot * cos(x4))
    x5dot = u

    f = Function('f', [x, u, t], [vertcat(x1dot, x2dot, x3dot, x4dot, x5dot)])

    x2dotdot_expr = x3dot * sin(x4) + x3 * x4dot * cos(x4) + whdot
    x2dotdot = Function('x2dotdot', [x, u, t], [x2dotdot_expr])

    # Integration
    def rk4_step(xk, uk, tk, dt):
        k1 = f(xk, uk, tk)
        k2 = f(xk + dt / 2 * k1, uk, tk + dt / 2)
        k3 = f(xk + dt / 2 * k2, uk, tk + dt / 2)
        k4 = f(xk + dt * k3, uk, tk + dt)
        return xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # npl prep
    w = []  # List of decision variables
    w0 = []  # Initial guess
    lbw = []  # Lower bounds
    ubw = []  # Upper bounds
    g = []  # Constraints
    lbg = []  # Lower bound on constraints
    ubg = []  # Upper bound on constraints
    J = 0  # Objective function

    x_initial_sym = MX.sym('x_initial', 5)

    # initial condition
    min_h = MX.sym('min_h')
    w += [min_h]
    lbw += [0]
    ubw += [inf]
    w0 += [600]

    Xk = MX.sym('X_0', 5)
    w += [Xk]
    lbw += [-inf] * 5
    ubw += [inf] * 5
    w0 += x_initial
    # Add constraint: Xk == x_initial_sym
    g += [Xk - x_initial_sym]
    lbg += [0] * 5
    ubg += [0] * 5

    Tk = 0

    # npl set up
    for k in range(params.nu):
        # control
        Uk = MX.sym('U_' + str(k))
        w += [Uk]
        lbw += [-3 * pi / 180]
        ubw += [3 * pi / 180]
        w0 += [0]

        # state integration
        Xk_end = rk4_step(Xk, Uk, Tk, params.dut)
        h_ddot_val = x2dotdot(Xk, Uk, Tk)

        Tk += params.dut

        # state
        Xk = MX.sym('X_' + str(k + 1), 5)
        w += [Xk]
        lbw += [-inf, 0, -inf, -inf, -inf]
        ubw += [inf, inf, inf, inf, 17.2 * pi / 180]
        w0 += [k * 10000 / params.nu, x_initial[1], x_initial[2], x_initial[3], x_initial[4]]
        g += [Xk_end - Xk]
        lbg += [0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0]

        g += [Xk[1] - min_h]
        lbg += [0]
        ubg += [inf]

        # Add to constraint list
        g += [h_ddot_val]
        lbg += [-2 * params.g]
        ubg += [10 * params.g]

    XF = 7.431 * pi / 180
    g += [Xk[3] - XF]
    lbg += [0]
    ubg += [0]

    J = -min_h

    opts = {'ipopt': {'print_level': 3}}
    npl = {'x': vertcat(*w), 'f': J, 'g': vertcat(*g), 'p': x_initial_sym}
    solver = nlpsol('solver', 'ipopt', npl, opts)

    x_initial_val = DM(x_initial)
    arg = {'x0': w0, 'lbx': lbw, 'ubx': ubw, 'lbg': lbg, 'ubg': ubg, 'p': x_initial_val}

    # Solve
    sol = solver(**arg)
    w_opt = sol['x'].full().flatten()
    J_opt = sol['f'].full().flatten()
    lam_g_opt = sol['lam_g'].full().flatten()
    lam_w_opt = sol['lam_x'].full().flatten()

    # ========== Sensitivity Analysis ==========
    g = vertcat(*g)
    w = vertcat(*w)
    # KKT variables: w and λ
    lam_g = MX.sym("lam", g.shape[0])
    lam_w = MX.sym("lam", w.shape[0])
    L = J + dot(lam_g, g) + dot(lam_w, w)  # Lagrangian

    # KKT conditions
    grad_L = gradient(L, w)
    comp_slack_g = diag(lam_g) @ g
    comp_slack_w = diag(lam_w) @ w

    # F(z,p)
    F = vertcat(grad_L, comp_slack_g, comp_slack_w)
    z = vertcat(w, lam_g, lam_w)

    # Compute Jacobians
    Fz = jacobian(F, z)
    Fp = jacobian(F, x_initial_sym)

    # Evaluate at solution
    z_val = np.concatenate([w_opt, lam_g_opt, lam_w_opt])  # z^*

    Fz_func = Function("Fz", [z, x_initial_sym], [Fz])
    Fp_func = Function("Fp", [z, x_initial_sym], [Fp])

    Fz_eval = Fz_func(z_val, x_initial_val)
    Fp_eval = Fp_func(z_val, x_initial_val)

    # Solve for dz/dp
    dzdp = -np.linalg.solve(Fz_eval, Fp_eval)
    dwdp = dzdp[0:w.shape[0]]

    return w_opt, J_opt, dwdp

def solver_min_h_scaled_sa_k_sym(k_value, N: int = params.nu):
    params.nu = N
    params.dut = params.tf / params.nu

    # State, Time and Control
    x1 = MX.sym('x1')  # x
    x2 = MX.sym('x2')  # h
    x3 = MX.sym('x3')  # V
    x4 = MX.sym('x4')  # gamma
    x5 = MX.sym('x5')  # alpha
    t = MX.sym('t')  # time
    u = MX.sym('u')  # control
    k_sym = MX.sym('k_sym')  # wind parameter
    x = vertcat(x1, x2, x3, x4, x5)

    # Wind
    A1 = -50 + params.a * x1 ** 3 + params.b * x1 ** 4
    A2 = 0.025 * (x1 - 2300)
    A3 = 50 - params.a * (4600 - x1) ** 3 - params.b * (4600 - x1) ** 4
    A4 = 50
    A = if_else(x1 <= 500, A1,
                if_else(x1 <= 4100, A2,
                        if_else(x1 <= 4600, A3, A4)))

    B1 = params.d * x1 ** 3 + params.e * x1 ** 4
    B2 = -51 * exp(-params.c * (x1 - 2300) ** 4)
    B3 = params.d * (4600 - x1) ** 3 + params.e * (4600 - x1) ** 4
    B4 = 0
    B = if_else(x1 <= 500, B1,
                if_else(x1 <= 4100, B2,
                        if_else(x1 <= 4600, B3, B4)))

    wind_x_expr = k_sym * A
    wind_x = Function('wind_x', [x,k_sym], [wind_x_expr])
    wind_h_expr = k_sym * x2 * B / params.h_star
    wind_h = Function('wind_h', [x,k_sym], [wind_h_expr])

    dWx_dx = Function("dWx_dx", [x,k_sym], [gradient(wind_x_expr, x1)])
    dWh_dx = Function("dWh_dx", [x,k_sym], [gradient(wind_h_expr, x1)])
    dWh_dh = Function("dWh_dh", [x,k_sym], [gradient(wind_h_expr, x2)])

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2

    # ode
    x1dot = x3 * cos(x4) + wind_x(x,k_sym)
    x2dot = x3 * sin(x4) + wind_h(x,k_sym)

    wxdot = dWx_dx(x,k_sym) * x1dot
    whdot = dWh_dx(x,k_sym) * x1dot + dWh_dh(x,k_sym) * x2dot
    x3_safe = fmax(x3, params.eps)

    x3dot = T / params.m * cos(x5 + params.delta) - D / params.m - params.g * sin(x4) - (
            wxdot * cos(x4) + whdot * sin(x4))
    x4dot = T / (params.m * x3) * sin(x5 + params.delta) + L / (params.m * x3) - params.g / x3_safe * cos(x4) + (
            1 / x3_safe) * (wxdot * sin(x4) - whdot * cos(x4))
    x5dot = u

    f_org = Function('f_org', [x, u, t, k_sym], [vertcat(x1dot, x2dot, x3dot, x4dot, x5dot)])

    x2dotdot_expr = x3dot * sin(x4) + x3 * x4dot * cos(x4) + whdot
    x2dotdot = Function('x2dotdot', [x, u, t, k_sym], [x2dotdot_expr])

    # scaled state, control and sys dynamics
    xs = MX.sym('xs', 5)
    us = MX.sym('us')
    f_s = params.scale_x * f_org(params.inv_scale_x * xs, params.inv_scale_u * us, t, k_sym)
    f_scaled = Function('f_scaled', [xs, us, t, k_sym], [f_s])

    # Integration
    def rk4_step(xsk, usk, tk, dt, k_val):
        k1 = f_scaled(xsk, usk, tk, k_val)
        k2 = f_scaled(xsk + dt / 2 * k1, usk, tk + dt / 2, k_val)
        k3 = f_scaled(xsk + dt / 2 * k2, usk, tk + dt / 2, k_val)
        k4 = f_scaled(xsk + dt * k3, usk, tk + dt, k_val)
        return xsk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # npl prep
    w = []  # List of decision variables
    w0 = []  # Initial guess
    lbw = []  # Lower bounds
    ubw = []  # Upper bounds
    g = []  # Constraints
    lbg = []  # Lower bound on constraints
    ubg = []  # Upper bound on constraints
    J = 0  # Objective function

    # initial condition
    min_h_s = MX.sym('min_h_s')
    w += [min_h_s]
    lbw += [0]
    ubw += [inf]
    w0 += [600*float(params.scale_h)]

    x_initial = [0, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
    xs_initial = [x_initial[i] * float(params.scale_x[i]) for i in range(5)]
    Xsk = MX.sym('Xs_0', 5)
    w += [Xsk]
    lbw += xs_initial
    ubw += xs_initial
    w0 += xs_initial

    Tk = 0

    # npl set up
    for k in range(params.nu):
        # control
        Usk = MX.sym('Us_' + str(k))
        w += [Usk]
        lbw += [-3 * pi / 180 * params.scale_u]
        ubw += [3 * pi / 180 * params.scale_u]
        w0 += [0]

        # state integration
        Xsk_end = rk4_step(Xsk, Usk, Tk, params.dut, k_sym)
        h_ddot_val = x2dotdot(Xsk*params.inv_scale_x, Usk*params.inv_scale_u, Tk, k_sym)

        Tk += params.dut

        # state
        Xsk = MX.sym('Xs_' + str(k + 1), 5)
        w += [Xsk]
        lbw += [-inf, 0, -inf, -inf, -inf]
        ubw += [inf, inf, inf, inf, 17.2 * pi / 180 * params.scale_x[4]]
        w0 += [k * 10000 / params.nu * params.scale_x[0], 600 * params.scale_x[1], 239.7* params.scale_x[2], -2.249 * pi / 180 * params.scale_x[3], 7.353 * pi / 180 * params.scale_x[4]]
        g += [Xsk_end - Xsk]
        lbg += [0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0]

        g += [Xsk[1] - min_h_s]
        lbg += [0]
        ubg += [inf]

        # Add to constraint list
        g += [h_ddot_val]
        lbg += [-2 * params.g]
        ubg += [10 * params.g]

    XF = 7.431 * pi / 180 * float(params.scale_x[3])
    g += [Xsk[3] - XF]
    lbg += [0]
    ubg += [0]

    J = -min_h_s*params.inv_scale_h

    opts = {'ipopt': { 'print_level': 3}}
    npl = {'x': vertcat(*w), 'f': J, 'g': vertcat(*g), 'p': k_sym}
    solver = nlpsol('solver', 'ipopt', npl, opts)

    arg = {'x0': vertcat(*w0), 'lbx': vertcat(*lbw), 'ubx': vertcat(*ubw), 'lbg': vertcat(*lbg), 'ubg': vertcat(*ubg), 'p':k_value}

    # Solve
    sol = solver(**arg)
    w_opt = sol['x'].full().flatten()
    J_opt = sol['f'].full().flatten()
    lam_g_opt = sol['lam_g'].full().flatten()
    lam_w_opt = sol['lam_x'].full().flatten()

    # ========== Sensitivity Analysis ==========
    g = vertcat(*g)
    w = vertcat(*w)
    # KKT variables: w and λ
    lam_g = MX.sym("lam", g.shape[0])
    lam_w = MX.sym("lam", w.shape[0])
    L = J + dot(lam_g, g) + dot(lam_w, w)  # Lagrangian

    # KKT conditions
    grad_L = gradient(L, w)
    comp_slack_g = diag(lam_g) @ g
    comp_slack_w = diag(lam_w) @ w

    # F(z,p)
    F = vertcat(grad_L, comp_slack_g, comp_slack_w)
    z = vertcat(w, lam_g, lam_w)

    # Compute Jacobians
    Fz = jacobian(F, z)
    Fp = jacobian(F,k_sym)

    # Evaluate at solution
    z_val = np.concatenate([w_opt, lam_g_opt, lam_w_opt])  # z^*

    Fz_func = Function("Fz", [z, k_sym], [Fz])
    Fp_func = Function("Fp", [z, k_sym], [Fp])

    Fz_eval = Fz_func(z_val, k_value)
    Fp_eval = Fp_func(z_val, k_value)

    # Solve for dz/dp
    dzdp = -np.linalg.lstsq(Fz_eval, Fp_eval, rcond=None)[0]
    dwdp = dzdp[0:w.shape[0]]

    return w_opt, J_opt, dwdp

def solver_min_h_scaled_sa_xi_sym(k_value, x_initial: list[float] = [0, 600, 239.7, -2.249 * np.pi / 180, 7.353 * np.pi / 180], N: int = params.nu):
    params.nu = N
    params.dut = params.tf / params.nu

    # State, Time and Control
    x1 = MX.sym('x1')  # x
    x2 = MX.sym('x2')  # h
    x3 = MX.sym('x3')  # V
    x4 = MX.sym('x4')  # gamma
    x5 = MX.sym('x5')  # alpha
    t = MX.sym('t')  # time
    u = MX.sym('u')  # control
    x = vertcat(x1, x2, x3, x4, x5)

    # Wind
    A1 = -50 + params.a * x1 ** 3 + params.b * x1 ** 4
    A2 = 0.025 * (x1 - 2300)
    A3 = 50 - params.a * (4600 - x1) ** 3 - params.b * (4600 - x1) ** 4
    A4 = 50
    A = if_else(x1 <= 500, A1,
                if_else(x1 <= 4100, A2,
                        if_else(x1 <= 4600, A3, A4)))

    B1 = params.d * x1 ** 3 + params.e * x1 ** 4
    B2 = -51 * exp(-params.c * (x1 - 2300) ** 4)
    B3 = params.d * (4600 - x1) ** 3 + params.e * (4600 - x1) ** 4
    B4 = 0
    B = if_else(x1 <= 500, B1,
                if_else(x1 <= 4100, B2,
                        if_else(x1 <= 4600, B3, B4)))

    wind_x_expr = k_value * A
    wind_x = Function('wind_x', [x1], [wind_x_expr])
    wind_h_expr = k_value * x2 * B / params.h_star
    wind_h = Function('wind_h', [x1, x2], [wind_h_expr])

    dWx_dx = Function("dWx_dx", [x1], [gradient(wind_x_expr, x1)])
    dWh_dx = Function("dWh_dx", [x1, x2], [gradient(wind_h_expr, x1)])
    dWh_dh = Function("dWh_dh", [x1, x2], [gradient(wind_h_expr, x2)])

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2

    # ode
    x1dot = x3 * cos(x4) + wind_x(x1)
    x2dot = x3 * sin(x4) + wind_h(x1, x2)

    wxdot = dWx_dx(x1) * x1dot
    whdot = dWh_dx(x1, x2) * x1dot + dWh_dh(x1, x2) * x2dot
    x3_safe = fmax(x3, params.eps)

    x3dot = T / params.m * cos(x5 + params.delta) - D / params.m - params.g * sin(x4) - (
            wxdot * cos(x4) + whdot * sin(x4))
    x4dot = T / (params.m * x3) * sin(x5 + params.delta) + L / (params.m * x3) - params.g / x3_safe * cos(x4) + (
            1 / x3_safe) * (wxdot * sin(x4) - whdot * cos(x4))
    x5dot = u

    f_org = Function('f_org', [x, u, t], [vertcat(x1dot, x2dot, x3dot, x4dot, x5dot)])

    x2dotdot_expr = x3dot * sin(x4) + x3 * x4dot * cos(x4) + whdot
    x2dotdot = Function('x2dotdot', [x, u, t], [x2dotdot_expr])

    # scaled state, control and sys dynamics
    xs = MX.sym('xs', 5)
    us = MX.sym('us')
    f_s = params.scale_x * f_org(params.inv_scale_x * xs, params.inv_scale_u * us, t)
    f_scaled = Function('f_scaled', [xs, us, t], [f_s])

    # Integration
    def rk4_step(xsk, usk, tk, dt):
        k1 = f_scaled(xsk, usk, tk)
        k2 = f_scaled(xsk + dt / 2 * k1, usk, tk + dt / 2)
        k3 = f_scaled(xsk + dt / 2 * k2, usk, tk + dt / 2)
        k4 = f_scaled(xsk + dt * k3, usk, tk + dt)
        return xsk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # npl prep
    w = []  # List of decision variables
    w0 = []  # Initial guess
    lbw = []  # Lower bounds
    ubw = []  # Upper bounds
    g = []  # Constraints
    lbg = []  # Lower bound on constraints
    ubg = []  # Upper bound on constraints
    J = 0  # Objective function

    xs_initial_sym = MX.sym('xs_initial', 5)

    # initial condition
    min_h_s = MX.sym('min_h_s')
    w += [min_h_s]
    lbw += [0]
    ubw += [inf]
    w0 += [500*float(params.scale_h)]

    xs_initial = [x_initial[i] * float(params.scale_x[i]) for i in range(5)]
    Xsk = MX.sym('Xs_0', 5)
    w += [Xsk]
    lbw += [-inf] * 5
    ubw += [inf] * 5
    w0 += x_initial
    # Add constraint: Xk == x_initial_sym
    g += [Xsk - xs_initial_sym]
    lbg += [0] * 5
    ubg += [0] * 5

    Tk = 0

    # npl set up
    for k in range(params.nu):
        # control
        Usk = MX.sym('Us_' + str(k))
        w += [Usk]
        lbw += [-3 * pi / 180 * params.scale_u]
        ubw += [3 * pi / 180 * params.scale_u]
        w0 += [0]

        # state integration
        Xsk_end = rk4_step(Xsk, Usk, Tk, params.dut)
        h_ddot_val = x2dotdot(Xsk*params.inv_scale_x, Usk*params.inv_scale_u, Tk)

        Tk += params.dut

        # state
        Xsk = MX.sym('Xs_' + str(k + 1), 5)
        w += [Xsk]
        lbw += [-inf, 0, -inf, -inf, -inf]
        ubw += [inf, inf, inf, inf, 17.2 * pi / 180 * params.scale_x[4]]
        w0 += [k * 10000 / params.nu * params.scale_x[0], 600 * params.scale_x[1], 239.7* params.scale_x[2], -2.249 * pi / 180 * params.scale_x[3], 7.353 * pi / 180 * params.scale_x[4]]
        g += [Xsk_end - Xsk]
        lbg += [0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0]

        g += [Xsk[1] - min_h_s]
        lbg += [0]
        ubg += [inf]

        # Add to constraint list
        g += [h_ddot_val]
        lbg += [-2 * params.g]
        ubg += [10 * params.g]

    XF = 7.431 * pi / 180 * float(params.scale_x[3])
    g += [Xsk[3] - XF]
    lbg += [0]
    ubg += [0]

    J = -min_h_s*params.inv_scale_h

    opts = {'ipopt': { 'print_level': 3}}
    npl = {'x': vertcat(*w), 'f': J, 'g': vertcat(*g), 'p': xs_initial_sym}
    solver = nlpsol('solver', 'ipopt', npl, opts)
    xs_initial_val = DM(xs_initial)
    arg = {'x0': vertcat(*w0), 'lbx': vertcat(*lbw), 'ubx': vertcat(*ubw), 'lbg': vertcat(*lbg), 'ubg': vertcat(*ubg), 'p': xs_initial_val}

    # Solve
    sol = solver(**arg)
    w_opt = sol['x'].full().flatten()
    J_opt = sol['f'].full().flatten()
    lam_g_opt = sol['lam_g'].full().flatten()
    lam_w_opt = sol['lam_x'].full().flatten()

    # ========== Sensitivity Analysis ==========
    g = vertcat(*g)
    w = vertcat(*w)
    # KKT variables: w and λ
    lam_g = MX.sym("lam", g.shape[0])
    lam_w = MX.sym("lam", w.shape[0])
    L = J + dot(lam_g, g) + dot(lam_w, w)  # Lagrangian

    # KKT conditions
    grad_L = gradient(L, w)
    comp_slack_g = diag(lam_g) @ g
    comp_slack_w = diag(lam_w) @ w

    # F(z,p)
    F = vertcat(grad_L, comp_slack_g, comp_slack_w)
    z = vertcat(w, lam_g, lam_w)

    # Compute Jacobians
    Fz = jacobian(F, z)
    Fp = jacobian(F, xs_initial_sym)

    # Evaluate at solution
    z_val = np.concatenate([w_opt, lam_g_opt, lam_w_opt])  # z^*

    Fz_func = Function("Fz", [z, xs_initial_sym], [Fz])
    Fp_func = Function("Fp", [z, xs_initial_sym], [Fp])

    Fz_eval = Fz_func(z_val, xs_initial_val)
    Fp_eval = Fp_func(z_val, xs_initial_val)

    # Solve for dz/dp
    dzdp = -np.linalg.solve(Fz_eval, Fp_eval)
    dwdp = dzdp[0:w.shape[0]]

    return w_opt, J_opt, dwdp

def plot_dwdxi(dwdp, N: int = params.nu):
    dt = params.tf/N
    time_points = np.arange(0, params.tf + dt, dt)

    sJ = -dwdp[0, :]

    sx1, sx2, sx3, sx4, sx5 = [], [], [], [], []
    su = []

    def get_stage_indices(k: int):

        base = 1 + k * 6
        x_indices = list(range(base, base + 5))
        u_index = base + 5 if k < N else None
        return x_indices, u_index

    for k in range(N + 1):
        x_inds, u_ind = get_stage_indices(k)

        sx1.append(dwdp[x_inds[0], :])  # x1_k
        sx2.append(dwdp[x_inds[1], :])  # x2_k
        sx3.append(dwdp[x_inds[2], :])  # x3_k
        sx4.append(dwdp[x_inds[3], :])  # x4_k
        sx5.append(dwdp[x_inds[4], :])  # x5_k

        if u_ind is not None:
            su.append(dwdp[u_ind, :])  # u_k

    sx1 = np.array(sx1)
    sx2 = np.array(sx2)
    sx3 = np.array(sx3)
    sx4 = np.array(sx4)
    sx5 = np.array(sx5)
    su = np.array(su)

    labels = [r'$\partial x_1 / \partial x_{1,initial}$',
              r'$\partial x_1 / \partial x_{2,initial}$',
              r'$\partial x_1 / \partial x_{3,initial}$',
              r'$\partial x_1 / \partial x_{4,initial}$',
              r'$\partial x_1 / \partial x_{5,initial}$']
    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.plot(time_points, sx1[:, i], label=labels[i])
    plt.xlabel('Time [s]')
    plt.ylabel('Sensitivity')
    plt.legend()
    plt.grid()
    plt.show()

    labels = [r'$\partial x_2 / \partial x_{1,initial}$',
              r'$\partial x_2 / \partial x_{2,initial}$',
              r'$\partial x_2 / \partial x_{3,initial}$',
              r'$\partial x_2 / \partial x_{4,initial}$',
              r'$\partial x_2 / \partial x_{5,initial}$']
    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.plot(time_points, sx2[:, i], label=labels[i])
    plt.xlabel('Time [s]')
    plt.ylabel('Sensitivity')
    plt.legend()
    plt.grid()
    plt.show()

    labels = [r'$\partial x_3 / \partial x_{1,initial}$',
              r'$\partial x_3 / \partial x_{2,initial}$',
              r'$\partial x_3 / \partial x_{3,initial}$',
              r'$\partial x_3 / \partial x_{4,initial}$',
              r'$\partial x_3 / \partial x_{5,initial}$']
    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.plot(time_points, sx3[:, i], label=labels[i])
    plt.xlabel('Time [s]')
    plt.ylabel('Sensitivity')
    plt.legend()
    plt.grid()
    plt.show()

    labels = [r'$\partial x_4 / \partial x_{1,initial}$',
              r'$\partial x_4 / \partial x_{2,initial}$',
              r'$\partial x_4 / \partial x_{3,initial}$',
              r'$\partial x_4 / \partial x_{4,initial}$',
              r'$\partial x_4 / \partial x_{5,initial}$']
    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.plot(time_points, sx4[:, i], label=labels[i])
    plt.xlabel('Time [s]')
    plt.ylabel('Sensitivity')
    plt.legend()
    plt.grid()
    plt.show()

    labels = [r'$\partial x_5 / \partial x_{1,initial}$',
              r'$\partial x_5 / \partial x_{2,initial}$',
              r'$\partial x_5 / \partial x_{3,initial}$',
              r'$\partial x_5 / \partial x_{4,initial}$',
              r'$\partial x_5 / \partial x_{5,initial}$']
    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.plot(time_points, sx5[:, i], label=labels[i])
    plt.xlabel('Time [s]')
    plt.ylabel('Sensitivity')
    plt.legend()
    plt.grid()
    plt.show()

    labels = [r'$\partial u / \partial x_{1,initial}$',
              r'$\partial u / \partial x_{2,initial}$',
              r'$\partial u / \partial x_{3,initial}$',
              r'$\partial u / \partial x_{4,initial}$',
              r'$\partial u / \partial x_{5,initial}$']
    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.plot(time_points[:-1], su[:, i], label=labels[i])
    plt.xlabel('Time [s]')
    plt.ylabel('Sensitivity')
    plt.legend()
    plt.grid()
    plt.show()

    print(sJ)

def plot_dwdk(w, dwdp, N: int = params.nu):
    dt = params.tf / N
    time_points = np.arange(0, params.tf + dt, dt)

    omega_opt = w[0]
    x1_opt = w[1::6]
    x2_opt = w[2::6]
    x3_opt = w[3::6]
    x4_opt = w[4::6]
    x5_opt = w[5::6]
    u_opt = w[6::6]

    sJ = -dwdp[0]
    sx1 = dwdp[1::6]
    sx2 = dwdp[2::6]
    sx3 = dwdp[3::6]
    sx4 = dwdp[4::6]
    sx5 = dwdp[5::6]
    su = dwdp[6::6]

    print(-omega_opt)
    print(sJ)

    labels = [r'$\partial x_1 / \partial k$',
              r'$\partial x_2 / \partial k$',
              r'$\partial x_3 / \partial k$',
              r'$\partial x_4 / \partial k$',
              r'$\partial x_5 / \partial k$',
              r'$\partial u / \partial k$']
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, sx1, label=labels[0])
    plt.plot(time_points, x1_opt, label='x1')
    plt.xlabel('Time [s]')
    plt.ylabel('Sensitivity')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(time_points, sx2, label=labels[1])
    plt.plot(time_points, x2_opt, label='x2')
    plt.xlabel('Time [s]')
    plt.ylabel('Sensitivity')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(time_points, sx3, label=labels[2])
    plt.plot(time_points, x3_opt, label='x3')
    plt.xlabel('Time [s]')
    plt.ylabel('Sensitivity')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(time_points, sx4, label=labels[3])
    plt.plot(time_points, x4_opt, label='x4')
    plt.xlabel('Time [s]')
    plt.ylabel('Sensitivity')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(time_points, sx5, label=labels[4])
    plt.plot(time_points, x5_opt, label='x5')
    plt.xlabel('Time [s]')
    plt.ylabel('Sensitivity')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(time_points[:-1], su, label=labels[5])
    plt.plot(time_points[:-1], u_opt, label='u')
    plt.xlabel('Time [s]')
    plt.ylabel('Sensitivity')
    plt.legend()
    plt.grid()
    plt.show()

def solver_min_h_scaled_sa_xi_sym_new(k_value, x_initial: list[float] = [0, 600, 239.7, -2.249 * np.pi / 180, 7.353 * np.pi / 180], N: int = params.nu):
    params.nu = N
    params.dut = params.tf / params.nu

    # State, Time and Control
    x1 = MX.sym('x1')  # x
    x2 = MX.sym('x2')  # h
    x3 = MX.sym('x3')  # V
    x4 = MX.sym('x4')  # gamma
    x5 = MX.sym('x5')  # alpha
    t = MX.sym('t')  # time
    u = MX.sym('u')  # control
    x = vertcat(x1, x2, x3, x4, x5)

    # Wind
    A1 = -50 + params.a * x1 ** 3 + params.b * x1 ** 4
    A2 = 0.025 * (x1 - 2300)
    A3 = 50 - params.a * (4600 - x1) ** 3 - params.b * (4600 - x1) ** 4
    A4 = 50
    A = if_else(x1 <= 500, A1,
                if_else(x1 <= 4100, A2,
                        if_else(x1 <= 4600, A3, A4)))

    B1 = params.d * x1 ** 3 + params.e * x1 ** 4
    B2 = -51 * exp(-params.c * (x1 - 2300) ** 4)
    B3 = params.d * (4600 - x1) ** 3 + params.e * (4600 - x1) ** 4
    B4 = 0
    B = if_else(x1 <= 500, B1,
                if_else(x1 <= 4100, B2,
                        if_else(x1 <= 4600, B3, B4)))

    wind_x_expr = k_value * A
    wind_x = Function('wind_x', [x1], [wind_x_expr])
    wind_h_expr = k_value * x2 * B / params.h_star
    wind_h = Function('wind_h', [x1, x2], [wind_h_expr])

    dWx_dx = Function("dWx_dx", [x1], [gradient(wind_x_expr, x1)])
    dWh_dx = Function("dWh_dx", [x1, x2], [gradient(wind_h_expr, x1)])
    dWh_dh = Function("dWh_dh", [x1, x2], [gradient(wind_h_expr, x2)])

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2

    # ode
    x1dot = x3 * cos(x4) + wind_x(x1)
    x2dot = x3 * sin(x4) + wind_h(x1, x2)

    wxdot = dWx_dx(x1) * x1dot
    whdot = dWh_dx(x1, x2) * x1dot + dWh_dh(x1, x2) * x2dot
    x3_safe = fmax(x3, params.eps)

    x3dot = T / params.m * cos(x5 + params.delta) - D / params.m - params.g * sin(x4) - (
            wxdot * cos(x4) + whdot * sin(x4))
    x4dot = T / (params.m * x3) * sin(x5 + params.delta) + L / (params.m * x3) - params.g / x3_safe * cos(x4) + (
            1 / x3_safe) * (wxdot * sin(x4) - whdot * cos(x4))
    x5dot = u

    f_org = Function('f_org', [x, u, t], [vertcat(x1dot, x2dot, x3dot, x4dot, x5dot)])

    x2dotdot_expr = x3dot * sin(x4) + x3 * x4dot * cos(x4) + whdot
    x2dotdot = Function('x2dotdot', [x, u, t], [x2dotdot_expr])

    # scaled state, control and sys dynamics
    xs = MX.sym('xs', 5)
    us = MX.sym('us')
    f_s = params.scale_x * f_org(params.inv_scale_x * xs, params.inv_scale_u * us, t)
    f_scaled = Function('f_scaled', [xs, us, t], [f_s])

    # Integration
    def rk4_step(xsk, usk, tk, dt):
        k1 = f_scaled(xsk, usk, tk)
        k2 = f_scaled(xsk + dt / 2 * k1, usk, tk + dt / 2)
        k3 = f_scaled(xsk + dt / 2 * k2, usk, tk + dt / 2)
        k4 = f_scaled(xsk + dt * k3, usk, tk + dt)
        return xsk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # npl prep
    w = []  # List of decision variables
    w0 = []  # Initial guess
    lbw = []  # Lower bounds
    ubw = []  # Upper bounds
    g = []  # Constraints
    lbg = []  # Lower bound on constraints
    ubg = []  # Upper bound on constraints
    J = 0  # Objective function

    xs_initial_sym = MX.sym('xs_initial', 5)

    # initial condition
    min_h_s = MX.sym('min_h_s')
    w += [min_h_s]
    lbw += [0]
    ubw += [inf]
    w0 += [500*float(params.scale_h)]

    xs_initial = [x_initial[i] * float(params.scale_x[i]) for i in range(5)]
    Xsk = MX.sym('Xs_0', 5)
    w += [Xsk]
    lbw += [-inf] * 5
    ubw += [inf] * 5
    w0 += x_initial
    # Add constraint: Xk == x_initial_sym
    g += [Xsk - xs_initial_sym]
    lbg += [0] * 5
    ubg += [0] * 5

    Tk = 0

    # npl set up
    for k in range(params.nu):
        # control
        Usk = MX.sym('Us_' + str(k))
        w += [Usk]
        lbw += [-3 * pi / 180 * params.scale_u]
        ubw += [3 * pi / 180 * params.scale_u]
        w0 += [0]

        # state integration
        Xsk_end = rk4_step(Xsk, Usk, Tk, params.dut)
        h_ddot_val = x2dotdot(Xsk*params.inv_scale_x, Usk*params.inv_scale_u, Tk)

        Tk += params.dut

        # state
        Xsk = MX.sym('Xs_' + str(k + 1), 5)
        w += [Xsk]
        lbw += [-inf, 0, -inf, -inf, -inf]
        ubw += [inf, inf, inf, inf, 17.2 * pi / 180 * params.scale_x[4]]
        w0 += [k * 10000 / params.nu * params.scale_x[0], 600 * params.scale_x[1], 239.7* params.scale_x[2], -2.249 * pi / 180 * params.scale_x[3], 7.353 * pi / 180 * params.scale_x[4]]
        g += [Xsk_end - Xsk]
        lbg += [0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0]

        g += [Xsk[1] - min_h_s]
        lbg += [0]
        ubg += [inf]

        # Add to constraint list
        g += [h_ddot_val]
        lbg += [-2 * params.g]
        ubg += [10 * params.g]

    g += [Xsk[3]]
    lbg += [0]
    ubg += [inf]

    J = -min_h_s*params.inv_scale_h

    opts = {'ipopt': { 'print_level': 3}}
    npl = {'x': vertcat(*w), 'f': J, 'g': vertcat(*g), 'p': xs_initial_sym}
    solver = nlpsol('solver', 'ipopt', npl, opts)
    xs_initial_val = DM(xs_initial)
    arg = {'x0': vertcat(*w0), 'lbx': vertcat(*lbw), 'ubx': vertcat(*ubw), 'lbg': vertcat(*lbg), 'ubg': vertcat(*ubg), 'p': xs_initial_val}

    # Solve
    sol = solver(**arg)
    w_opt = sol['x'].full().flatten()
    J_opt = sol['f'].full().flatten()
    lam_g_opt = sol['lam_g'].full().flatten()
    lam_w_opt = sol['lam_x'].full().flatten()

    # ========== Sensitivity Analysis ==========
    g = vertcat(*g)
    w = vertcat(*w)
    # KKT variables: w and λ
    lam_g = MX.sym("lam", g.shape[0])
    lam_w = MX.sym("lam", w.shape[0])
    L = J + dot(lam_g, g) + dot(lam_w, w)  # Lagrangian

    # KKT conditions
    grad_L = gradient(L, w)
    comp_slack_g = diag(lam_g) @ g
    comp_slack_w = diag(lam_w) @ w

    # F(z,p)
    F = vertcat(grad_L, comp_slack_g, comp_slack_w)
    z = vertcat(w, lam_g, lam_w)

    # Compute Jacobians
    Fz = jacobian(F, z)
    Fp = jacobian(F, xs_initial_sym)

    # Evaluate at solution
    z_val = np.concatenate([w_opt, lam_g_opt, lam_w_opt])  # z^*

    Fz_func = Function("Fz", [z, xs_initial_sym], [Fz])
    Fp_func = Function("Fp", [z, xs_initial_sym], [Fp])

    Fz_eval = Fz_func(z_val, xs_initial_val)
    Fp_eval = Fp_func(z_val, xs_initial_val)

    # Solve for dz/dp
    dzdp = -np.linalg.lstsq(Fz_eval, Fp_eval, rcond=None)[0]
    dwdp = dzdp[0:w.shape[0]]

    return w_opt, J_opt, dwdp
'''

