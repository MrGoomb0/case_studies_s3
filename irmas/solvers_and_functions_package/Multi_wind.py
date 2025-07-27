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
    scale_Q = 1/10**(19)
    inv_scale_Q = 10**(19)


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




# ---------- Integrators ----------

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

def MC_rk4_step(dynamics, xk, uk, tk, dt, k_value, s_value):
    k1, _ = dynamics(xk, uk, tk, k_value, s_value)
    k2, _ = dynamics(xk + dt / 2 * k1, uk, tk + dt / 2, k_value, s_value)
    k3, _ = dynamics(xk + dt / 2 * k2, uk, tk + dt / 2, k_value, s_value)
    k4, x2dotdot = dynamics(xk + dt * k3, uk, tk + dt, k_value, s_value)
    return xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), x2dotdot



# ---------- Bolza -----------
def solver_bolza_EJ_scaled(k_values, s_values, A_w, B_w, N: int = params.nu, x_initial=None, pesch_end_cond: bool=False, integrator = rk4_step_bolza, tol=1e-10,constr_viol_tol=1e-6):
    params.nu = N
    params.dut = params.tf / params.nu

    nk = len(k_values)

    # State, Time and Control
    x1 = MX.sym('x1')  # x
    x2 = MX.sym('x2')  # h
    x3 = MX.sym('x3')  # V
    x4 = MX.sym('x4')  # gamma
    x5 = MX.sym('x5')  # alpha
    t = MX.sym('t')  # time
    u = MX.sym('u')  # control
    x = vertcat(x1, x2, x3, x4, x5)

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2


    # npl prep
    w = []  # List of decision variables
    w0 = []  # Initial guess
    lbw = []  # Lower bounds
    ubw = []  # Upper bounds
    g = []  # Constraints
    lbg = []  # Lower bound on constraints
    ubg = []  # Upper bound on constraints
    Js = [0]*nk  # Objective function

    # control
    Us = MX.sym('Us', N)
    w += [Us]
    lbw += [-3 * pi / 180 * params.scale_u] * params.nu
    ubw += [3 * pi / 180 * params.scale_u] * params.nu
    w0 += [0] * params.nu

    for i in range(nk):
        k_value = k_values[i]
        s_value = s_values[i]

        # Wind
        wind_x_expr = k_value * A_w(x1, s_value)
        wind_x = Function('wind_x', [x1], [wind_x_expr])
        wind_h_expr = k_value * x2 * B_w(x1, s_value) / params.h_star
        wind_h = Function('wind_h', [x1, x2], [wind_h_expr])

        dWx_dx = Function("dWx_dx", [x1], [gradient(wind_x_expr, x1)])
        dWh_dx = Function("dWh_dx", [x1, x2], [gradient(wind_h_expr, x1)])
        dWh_dh = Function("dWh_dh", [x1, x2], [gradient(wind_h_expr, x2)])

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
        qf_s = params.scale_Q * qf_org(params.inv_scale_x * xs)
        qf_scaled = Function('qf_scaled', [xs], [qf_s])

        # initial condition
        if x_initial is None:
            x_initial = [0, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
        xs_initial = [x_initial[i] * float(params.scale_x[i]) for i in range(5)]
        Xsk = MX.sym('Xs_0' + str(i), 5)
        w += [Xsk]
        lbw += xs_initial
        ubw += xs_initial
        w0 += xs_initial

        Tk = 0

        # npl set up
        for k in range(params.nu):
            # control
            Usk = Us[k]

            # state integration
            Xsk_end, Js_end = integrator(f_scaled,qf_scaled, Xsk, Usk, Tk, params.dut, Js[i])
            Js[i] = Js_end
            h_ddot_val = x2dotdot(Xsk * params.inv_scale_x, Usk * params.inv_scale_u, Tk)

            Tk += params.dut

            # state
            Xsk = MX.sym('Xs_' + str(k + 1) +str(i), 5)
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

        g += [Xsk[3]]
        lbg += [0]
        ubg += [inf]

    ev=0
    for i in range(nk):
        ev = ev+Js[i]
    ev = ev/nk
    va = 0
    for i in range(nk):
        va = va+(Js[i]-ev)**2
    if nk>1:
        va = va/(nk-1)
    J = ev+va
    opts = {'ipopt': {'print_level': 3, 'tol': tol, 'constr_viol_tol': constr_viol_tol}}

    npl = {'x': vertcat(*w), 'f': J, 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', npl, opts)

    arg = {'x0': vertcat(*w0), 'lbx': vertcat(*lbw), 'ubx': vertcat(*ubw), 'lbg': vertcat(*lbg), 'ubg': vertcat(*ubg)}

    # Solve
    sol = solver(**arg)
    w_opt = sol['x'].full().flatten()
    J_opt = sol['f'].full().flatten()

    return w_opt, J_opt

def solver_bolza_Eh_scaled(k_values, s_values, A_w, B_w, N: int = params.nu, x_initial=None, pesch_end_cond: bool=False, integrator = rk4_step, tol=1e-10,constr_viol_tol=1e-6):
    params.nu = N
    params.dut = params.tf / params.nu

    nk = len(k_values)

    # State, Time and Control
    x1 = MX.sym('x1')  # x
    x2 = MX.sym('x2')  # h
    x3 = MX.sym('x3')  # V
    x4 = MX.sym('x4')  # gamma
    x5 = MX.sym('x5')  # alpha
    t = MX.sym('t')  # time
    u = MX.sym('u')  # control
    x = vertcat(x1, x2, x3, x4, x5)

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2


    # npl prep
    w = []  # List of decision variables
    w0 = []  # Initial guess
    lbw = []  # Lower bounds
    ubw = []  # Upper bounds
    g = []  # Constraints
    lbg = []  # Lower bound on constraints
    ubg = []  # Upper bound on constraints
    Js = 0
    h_list_list = []

    # control
    Us = MX.sym('Us', N)
    w += [Us]
    lbw += [-3 * pi / 180 * params.scale_u] * params.nu
    ubw += [3 * pi / 180 * params.scale_u] * params.nu
    w0 += [0] * params.nu

    for i in range(nk):
        k_value = k_values[i]
        s_value = s_values[i]
        h_list = []

        # Wind
        wind_x_expr = k_value * A_w(x1, s_value)
        wind_x = Function('wind_x', [x1], [wind_x_expr])
        wind_h_expr = k_value * x2 * B_w(x1, s_value) / params.h_star
        wind_h = Function('wind_h', [x1, x2], [wind_h_expr])

        dWx_dx = Function("dWx_dx", [x1], [gradient(wind_x_expr, x1)])
        dWh_dx = Function("dWh_dx", [x1, x2], [gradient(wind_h_expr, x1)])
        dWh_dh = Function("dWh_dh", [x1, x2], [gradient(wind_h_expr, x2)])

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

        # initial condition
        if x_initial is None:
            x_initial = [0, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
        xs_initial = [x_initial[i] * float(params.scale_x[i]) for i in range(5)]
        Xsk = MX.sym('Xs_0' + str(i), 5)
        w += [Xsk]
        lbw += xs_initial
        ubw += xs_initial
        w0 += xs_initial

        Tk = 0

        # npl set up
        for k in range(params.nu):
            h_list.append(Xsk[1])
            # control
            Usk = Us[k]

            # state integration
            Xsk_end = integrator(f_scaled, Xsk, Usk, Tk, params.dut)
            h_ddot_val = x2dotdot(Xsk * params.inv_scale_x, Usk * params.inv_scale_u, Tk)

            Tk += params.dut

            # state
            Xsk = MX.sym('Xs_' + str(k + 1) +str(i), 5)
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

        g += [Xsk[3]]
        lbg += [0]
        ubg += [inf]

        h_list_list.append(h_list)

    # objective function

    for k in range(params.nu):
        ev = 0
        for i in range(nk):
            ev = ev+h_list_list[i][k]*params.inv_scale_h
        ev=ev/nk
        va = 0
        for i in range(nk):
            va = va+(h_list_list[i][k]*params.inv_scale_h-ev)**2
        if nk>1:
            va/(nk-1)
        Js = Js + (params.hR - ev + va*0.01) ** params.q * params.dut

    Js= Js*params.scale_Q

    opts = {'ipopt': {'print_level': 3, 'tol': tol, 'constr_viol_tol': constr_viol_tol}}

    npl = {'x': vertcat(*w), 'f': Js, 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', npl, opts)

    arg = {'x0': vertcat(*w0), 'lbx': vertcat(*lbw), 'ubx': vertcat(*ubw), 'lbg': vertcat(*lbg), 'ubg': vertcat(*ubg)}

    # Solve
    sol = solver(**arg)
    w_opt = sol['x'].full().flatten()
    J_opt = sol['f'].full().flatten()

    return w_opt, J_opt

def solver_bolza_Eh_geuss_scaled(k_values, s_values, A_w, B_w, x_geuss: list[list[float]], u_geuss: list[float], N: int = params.nu, x_initial=None, pesch_end_cond: bool=False, integrator = rk4_step, tol=1e-10,constr_viol_tol=1e-6):
    params.nu = N
    params.dut = params.tf / params.nu

    nk = len(k_values)

    # State, Time and Control
    x1 = MX.sym('x1')  # x
    x2 = MX.sym('x2')  # h
    x3 = MX.sym('x3')  # V
    x4 = MX.sym('x4')  # gamma
    x5 = MX.sym('x5')  # alpha
    t = MX.sym('t')  # time
    u = MX.sym('u')  # control
    x = vertcat(x1, x2, x3, x4, x5)

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2


    # npl prep
    w = []  # List of decision variables
    w0 = []  # Initial guess
    lbw = []  # Lower bounds
    ubw = []  # Upper bounds
    g = []  # Constraints
    lbg = []  # Lower bound on constraints
    ubg = []  # Upper bound on constraints
    Js = 0
    h_list_list = []

    # control
    Us = MX.sym('Us', N)
    w += [Us]
    lbw += [-3 * pi / 180 * params.scale_u] * params.nu
    ubw += [3 * pi / 180 * params.scale_u] * params.nu
    w0.extend(u_geuss)

    for i in range(nk):
        k_value = k_values[i]
        s_value = s_values[i]
        h_list = []

        # Wind
        wind_x_expr = k_value * A_w(x1, s_value)
        wind_x = Function('wind_x', [x1], [wind_x_expr])
        wind_h_expr = k_value * x2 * B_w(x1, s_value) / params.h_star
        wind_h = Function('wind_h', [x1, x2], [wind_h_expr])

        dWx_dx = Function("dWx_dx", [x1], [gradient(wind_x_expr, x1)])
        dWh_dx = Function("dWh_dx", [x1, x2], [gradient(wind_h_expr, x1)])
        dWh_dh = Function("dWh_dh", [x1, x2], [gradient(wind_h_expr, x2)])

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

        # initial condition
        if x_initial is None:
            x_initial = [0, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
        xs_initial = [x_initial[i] * float(params.scale_x[i]) for i in range(5)]
        Xsk = MX.sym('Xs_0' + str(i), 5)
        w += [Xsk]
        lbw += xs_initial
        ubw += xs_initial
        w0 += xs_initial

        Tk = 0

        # npl set up
        for k in range(params.nu):
            h_list.append(Xsk[1])
            # control
            Usk = Us[k]

            # state integration
            Xsk_end = integrator(f_scaled, Xsk, Usk, Tk, params.dut)
            h_ddot_val = x2dotdot(Xsk * params.inv_scale_x, Usk * params.inv_scale_u, Tk)

            Tk += params.dut

            # state
            Xsk = MX.sym('Xs_' + str(k + 1) +str(i), 5)
            w += [Xsk]
            lbw += [-inf, 0, -inf, -inf, -17.2 * pi / 180 * params.scale_x[4]]
            ubw += [inf, inf, inf, inf, 17.2 * pi / 180 * params.scale_x[4]]
            w0 += [x_geuss[0][k], x_geuss[1][k], x_geuss[2][k], x_geuss[3][k], x_geuss[4][k]]
            g += [Xsk_end - Xsk]
            lbg += [0, 0, 0, 0, 0]
            ubg += [0, 0, 0, 0, 0]

            # Add to constraint list
            g += [h_ddot_val]
            lbg += [-2 * params.g]
            ubg += [10 * params.g]

        g += [Xsk[3]]
        lbg += [0]
        ubg += [inf]

        h_list_list.append(h_list)

    # objective function

    for k in range(params.nu):
        ev = 0
        for i in range(nk):
            ev = ev+h_list_list[i][k]*params.inv_scale_h
        ev=ev/nk
        va = 0
        for i in range(nk):
            va = va+(h_list_list[i][k]*params.inv_scale_h-ev)**2
        if nk>1:
            va/(nk-1)
        Js = Js + (params.hR - ev + va*0.01) ** params.q * params.dut

    Js= Js*params.scale_Q

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
def solver_min_h_scaled(k_values, s_values, A_w, B_w, N: int = params.nu, x_initial=None, integrator = rk4_step, tol=1e-10, constr_viol_tol=1e-6):
    params.nu = N
    params.dut = params.tf / params.nu

    nk = len(k_values)  # Number of k:s

    # State, Time and Control
    x1 = MX.sym('x1')  # x
    x2 = MX.sym('x2')  # h
    x3 = MX.sym('x3')  # V
    x4 = MX.sym('x4')  # gamma
    x5 = MX.sym('x5')  # alpha
    t = MX.sym('t')  # time
    u = MX.sym('u')  # control
    x = vertcat(x1, x2, x3, x4, x5)

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2

    # npl prep
    w = []  # List of decision variables
    w0 = []  # Initial guess
    lbw = []  # Lower bounds
    ubw = []  # Upper bounds
    g = []  # Constraints
    lbg = []  # Lower bound on constraints
    ubg = []  # Upper bound on constraints

    # control
    Us = MX.sym('Us', N)
    w += [Us]
    lbw += [-3 * pi / 180 * params.scale_u]*params.nu
    ubw += [3 * pi / 180 * params.scale_u]*params.nu
    w0 += [0] * params.nu

    # initial condition
    min_h_s = MX.sym('min_h_s')
    w += [min_h_s]
    lbw += [0]
    ubw += [inf]
    w0 += [600 * float(params.scale_h)]


    for i in range(nk):
        k_value =  k_values[i]
        s_value = s_values[i]

        # Wind
        wind_x_expr = k_value * A_w(x1, s_value)
        wind_x = Function('wind_x', [x1], [wind_x_expr])
        wind_h_expr = k_value * x2 * B_w(x1, s_value) / params.h_star
        wind_h = Function('wind_h', [x1, x2], [wind_h_expr])

        dWx_dx = Function("dWx_dx", [x1], [gradient(wind_x_expr, x1)])
        dWh_dx = Function("dWh_dx", [x1, x2], [gradient(wind_h_expr, x1)])
        dWh_dh = Function("dWh_dh", [x1, x2], [gradient(wind_h_expr, x2)])
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

        if x_initial is None:
            x_initial = [0, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
        xs_initial = [x_initial[j] * float(params.scale_x[j]) for j in range(5)]
        Xsk = MX.sym('Xs_0' + str(i), 5)
        w += [Xsk]
        lbw += xs_initial
        ubw += xs_initial
        w0 += xs_initial

        Tk = 0

        for k in range(params.nu):
            Usk = Us[k]
            # state integration
            Xsk_end = integrator(f_scaled, Xsk, Usk, Tk, params.dut)
            h_ddot_val = x2dotdot(Xsk*params.inv_scale_x, Usk*params.inv_scale_u, Tk)

            Tk += params.dut

            # state
            Xsk = MX.sym('Xs_' + str(k + 1) + str(i), 5)
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



# --------- other ---------
def u_opt_return(w_opt: list[float], N: int, is_scaled: bool):
    # returns the optimal control from the result of the solver
    u_opt = w_opt[:N]
    if is_scaled:
        u_opt = [x * float(params.inv_scale_u) for x in u_opt]
    return u_opt

def ploter(w_opt: list[float], k_values: list[float], s_values: list[float], N: int, is_scaled: bool, is_bolza:bool):

    tf = params.tf
    nk = len(k_values)
    u_opt = w_opt[:N]
    if is_bolza:
        w = w_opt[N:]  # only states
    else:
        h_min = w_opt[N]*params.inv_scale_h
        w = w_opt[N + 1:] # only states

    # going over all k values
    all_lists = []
    for i in range(nk):
        new_w = w[:N * 5 + 5]
        all_lists.append(new_w)
        w = w[N * 5 + 5:]

    tgrid = [tf / N * k for k in range(N + 1)]
    plt.figure()
    plt.clf()
    for i in range(nk):
        x1_opt = all_lists[i][0::5]
        x2_opt = all_lists[i][1::5]
        x3_opt = all_lists[i][2::5]
        x4_opt = all_lists[i][3::5]
        x5_opt = all_lists[i][4::5]

        if is_scaled:
            x1_opt = [x * float(params.inv_scale_x[0]) for x in x1_opt]
            x2_opt = [x * float(params.inv_scale_x[1]) for x in x2_opt]
            x3_opt = [x * float(params.inv_scale_x[2]) for x in x3_opt]
            x4_opt = [x * float(params.inv_scale_x[3]) for x in x4_opt]
            x5_opt = [x * float(params.inv_scale_x[4]) for x in x5_opt]
            u_opt = [x * float(params.inv_scale_u) for x in u_opt]

        print(min(x2_opt))

        plt.plot(x1_opt, x2_opt, '--', label=rf"k={k_values[i]} s={s_values[i]}")
    plt.grid()
    plt.xlabel('horizontal distance [ft]')
    plt.ylabel('altitude [ft]')
    plt.legend()
    plt.show()

    plt.figure()
    plt.clf()
    plt.step(tgrid[:-1], u_opt, where='post')
    plt.xlabel('t')
    plt.title('Optimal Control')
    plt.grid()
    plt.show()

    return

def sol_dic(w_opt: list[float], k_values: list[float], N: int, is_scaled: bool, is_bolza:bool):

    tf = params.tf
    nk = len(k_values)
    u_opt = w_opt[:N]
    if is_bolza:
        w = w_opt[N:]  # only states
    else:
        h_min = w_opt[N]*params.inv_scale_h
        w = w_opt[N + 1:] # only states

    # going over all k values
    all_lists = []
    for i in range(nk):
        new_w = w[:N * 5 + 5]
        all_lists.append(new_w)
        w = w[N * 5 + 5:]

    solsx1 = []
    solsx2 = []
    solsx3 = []
    solsx4 = []
    solsx5= []
    tgrid = [tf / N * k for k in range(N + 1)]

    for i in range(nk):
        x1_opt = all_lists[i][0::5]
        x2_opt = all_lists[i][1::5]
        x3_opt = all_lists[i][2::5]
        x4_opt = all_lists[i][3::5]
        x5_opt = all_lists[i][4::5]

        if is_scaled:
            x1_opt = [x * float(params.inv_scale_x[0]) for x in x1_opt]
            x2_opt = [x * float(params.inv_scale_x[1]) for x in x2_opt]
            x3_opt = [x * float(params.inv_scale_x[2]) for x in x3_opt]
            x4_opt = [x * float(params.inv_scale_x[3]) for x in x4_opt]
            x5_opt = [x * float(params.inv_scale_x[4]) for x in x5_opt]
            u_opt = [x * float(params.inv_scale_u) for x in u_opt]

        solsx1.append(x1_opt)
        solsx2.append(x2_opt)
        solsx3.append(x3_opt)
        solsx4.append(x4_opt)
        solsx5.append(x5_opt)

    return {'x': solsx1, 'h': solsx2, 'v': solsx3, 'gamma': solsx4, 'alpha': solsx5, 'u': u_opt, 't_grid':tgrid}

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





def simulate_trajectories(u_opt: list[float], M: int, k_mid: float, plot=True):

    def wind_fields(x1, x2, k):
        if x1 <= 500:
            A = -50 + params.a * x1 ** 3 + params.b * x1 ** 4
        elif x1 <= 4100:
            A = 0.025 * (x1 - 2300)
        elif x1 <= 4600:
            A = 50 - params.a * (4600 - x1) ** 3 - params.b * (4600 - x1) ** 4
        else:
            A = 50
        wx = k * A

        if x1 <= 500:
            B = params.d * x1 ** 3 + params.e * x1 ** 4
        elif x1 <= 4100:
            B = -51 * np.exp(-params.c * (x1 - 2300) ** 4)
        elif x1 <= 4600:
            B = params.d * (4600 - x1) ** 3 + params.e * (4600 - x1) ** 4
        else:
            B = 0
        wh = k * x2 * B / params.h_star

        dWx_dx = k * (
            3 * params.a * x1 ** 2 + 4 * params.b * x1 ** 3 if x1 <= 500 else
            0.025 if x1 <= 4100 else
            k * (3 * params.a * (4600 - x1) ** 2 + 4 * params.b * (4600 - x1) ** 3) if x1 <= 4600 else 0
        )
        dWh_dx = 0
        dWh_dh = 0
        if x1 <= 500:
            dWh_dx = k * x2 * (3 * params.d * x1 ** 2 + 4 * params.e * x1 ** 3) / params.h_star
            dWh_dh = k * B / params.h_star
        elif x1 <= 4100:
            dWh_dx = k * x2 * (
                204 * params.c * (x1 - 2300) ** 3 * np.exp(min(-params.c * (x1 - 2300) ** 4, 30))
            ) / params.h_star
            dWh_dh = k * B / params.h_star
        elif x1 <= 4600:
            dWh_dx = -k * x2 * (3 * params.d * (4600 - x1) ** 2 + 4 * params.e * (4600 - x1) ** 3) / params.h_star
            dWh_dh = k * B / params.h_star

        return wx, wh, dWx_dx, dWh_dx, dWh_dh

    def dynamics(x, u, t, k):
        x1, x2, x3, x4, x5 = x
        wx, wh, dWx_dx, dWh_dx, dWh_dh = wind_fields(x1, x2, k)

        if x5 > params.alpha_star:
            C_L = params.C0 + params.C1 * x5
        else:
            C_L = params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2

        beta = params.beta0 + params.beta_dot0 * t if t < params.sigma else 1.0
        T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)
        D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2
        L = 0.5 * params.rho * params.S * C_L * x3 ** 2

        x1dot = x3 * np.cos(x4) + wx
        x2dot = x3 * np.sin(x4) + wh
        wxdot = dWx_dx * x1dot
        whdot = dWh_dx * x1dot + dWh_dh * x2dot
        x3_safe = max(x3, params.eps)
        x3dot = (T * np.cos(x5 + params.delta) - D) / params.m - params.g * np.sin(x4) - \
                (wxdot * np.cos(x4) + whdot * np.sin(x4))
        x4dot = (T * np.sin(x5 + params.delta)) / (params.m * x3_safe) + L / (params.m * x3_safe) - \
                params.g * np.cos(x4) / x3_safe + (wxdot * np.sin(x4) - whdot * np.cos(x4)) / x3_safe
        x5dot = u
        return np.array([x1dot, x2dot, x3dot, x4dot, x5dot])

    def rk4(x, u, t, dt, k):
        k1 = dynamics(x, u, t, k)
        k2 = dynamics(x + dt / 2 * k1, u, t + dt / 2, k)
        k3 = dynamics(x + dt / 2 * k2, u, t + dt / 2, k)
        k4 = dynamics(x + dt * k3, u, t + dt, k)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    min_hs = []
    k_samples = np.random.normal(loc=k_mid, scale=0.25, size=M)
    violating_ks = {
        'x2dotdot_low': [],
        'x2dotdot_high': []
    }

    if plot:
        plt.figure()

    for k_val in k_samples:
        xk = np.array([0, 600, 239.7, -2.249 * np.pi / 180, 7.353 * np.pi / 180])
        t = 0
        traj = [xk.copy()]
        violated_low = False
        violated_high = False

        for i in range(params.nu):
            xk = rk4(xk, u_opt[i], t, params.dut, k_val)
            traj.append(xk.copy())
            t += params.dut

            x1, x2, x3, x4, x5 = xk
            wx, wh, dWx_dx, dWh_dx, dWh_dh = wind_fields(x1, x2, k_val)
            x1dot = x3 * np.cos(x4) + wx
            x2dot = x3 * np.sin(x4) + wh
            wxdot = dWx_dx * x1dot
            whdot = dWh_dx * x1dot + dWh_dh * x2dot
            x3_safe = max(x3, params.eps)

            T = params.beta0 * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)
            D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2
            L = 0.5 * params.rho * params.S * (params.C0 + params.C1 * x5) * x3 ** 2

            x3dot = (T * np.cos(x5 + params.delta) - D) / params.m - params.g * np.sin(x4) - \
                    (wxdot * np.cos(x4) + whdot * np.sin(x4))
            x4dot = (T * np.sin(x5 + params.delta)) / (params.m * x3_safe) + L / (params.m * x3_safe) - \
                    params.g * np.cos(x4) / x3_safe + (wxdot * np.sin(x4) - whdot * np.cos(x4)) / x3_safe
            x2dotdot = x3dot * np.sin(x4) + x3 * x4dot * np.cos(x4) + whdot

            if x2dotdot < -2 * params.g:
                violated_low = True
            if x2dotdot > 10 * params.g:
                violated_high = True

        if violated_low:
            violating_ks['x2dotdot_low'].append(float(k_val))
        if violated_high:
            violating_ks['x2dotdot_high'].append(float(k_val))

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
    print('Minimum altitude over all runs: ' + str(f"{min(min_hs):.3f}"))

    M = len(min_hs)
    crash_0 = math.fsum(h <= 0 for h in min_hs) / M * 100
    fail_50 = math.fsum(h <= 50 for h in min_hs) / M * 100
    fail_100 = math.fsum(h <= 100 for h in min_hs) / M * 100

    print('Crash rate: ' + str(crash_0) + '%')
    print('Fail 50 feet rate: ' + str(fail_50) + '%')
    print('Fail 100 rate: ' + str(fail_100) + '%')

    return violating_ks


def simulate_trajectories_smooth(u_opt: list[float], M: int, k_mid: float, plot=True):
    def Smooth(x1, x_start, x_end):
        t_smooth = (x1 - x_start) / (x_end - x_start + params.eps)
        return np.where(x1 < x_start, 0,
                        np.where(x1 > x_end, 1, 6 * t_smooth ** 5 - 15 * t_smooth ** 4 + 10 * t_smooth ** 3))

    def wind_fields(x1, x2, k):
        A1 = -50 + params.a * x1 ** 3 + params.b * x1 ** 4
        A2 = 0.025 * (x1 - 2300)
        A3 = 50 - params.a * (4600 - x1) ** 3 - params.b * (4600 - x1) ** 4
        A4 = 50
        s1 = Smooth(x1, 480, 520)
        s2 = Smooth(x1, 4080, 4120)
        s3 = Smooth(x1, 4580, 4620)
        A = np.where(x1 <= 500, (1 - s1) * A1 + s1 * A2,
                     np.where(x1 <= 4100, (1 - s2) * A2 + s2 * A3,
                              np.where(x1 <= 4600, (1 - s3) * A3 + s3 * A4, A4)))
        wx = k * A

        B1 = params.d * x1 ** 3 + params.e * x1 ** 4
        B2 = -51 * np.exp(-params.c * (x1 - 2300) ** 4)
        B3 = params.d * (4600 - x1) ** 3 + params.e * (4600 - x1) ** 4
        B4 = 0
        s1 = Smooth(x1, 480, 520)
        s2 = Smooth(x1, 4080, 4120)
        s3 = Smooth(x1, 4580, 4620)
        B = np.where(x1 <= 500, (1 - s1) * B1 + s1 * B2,
                     np.where(x1 <= 4100, (1 - s2) * B2 + s2 * B3,
                              np.where(x1 <= 4600, (1 - s3) * B3 + s3 * B4, B4)))
        wh = k * x2 * B / params.h_star

        # Derivatives: you may keep them piecewise or update similarly for smoothness
        dWx_dx = k * (
            3 * params.a * x1 ** 2 + 4 * params.b * x1 ** 3 if x1 <= 500 else
            0.025 if x1 <= 4100 else
            k * (3 * params.a * (4600 - x1) ** 2 + 4 * params.b * (4600 - x1) ** 3) if x1 <= 4600 else 0
        )
        dWh_dx = 0
        dWh_dh = 0
        if x1 <= 500:
            dWh_dx = k * x2 * (3 * params.d * x1 ** 2 + 4 * params.e * x1 ** 3) / params.h_star
            dWh_dh = k * B / params.h_star
        elif x1 <= 4100:
            dWh_dx = k * x2 * (
                    204 * params.c * (x1 - 2300) ** 3 * np.exp(min(-params.c * (x1 - 2300) ** 4, 30))
            ) / params.h_star
            dWh_dh = k * B / params.h_star
        elif x1 <= 4600:
            dWh_dx = -k * x2 * (3 * params.d * (4600 - x1) ** 2 + 4 * params.e * (4600 - x1) ** 3) / params.h_star
            dWh_dh = k * B / params.h_star

        return wx, wh, dWx_dx, dWh_dx, dWh_dh

    def dynamics(x, u, t, k):
        x1, x2, x3, x4, x5 = x
        wx, wh, dWx_dx, dWh_dx, dWh_dh = wind_fields(x1, x2, k)

        if x5 > params.alpha_star:
            C_L = params.C0 + params.C1 * x5
        else:
            C_L = params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2

        beta = params.beta0 + params.beta_dot0 * t if t < params.sigma else 1.0
        T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)
        D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2
        L = 0.5 * params.rho * params.S * C_L * x3 ** 2

        x1dot = x3 * np.cos(x4) + wx
        x2dot = x3 * np.sin(x4) + wh
        wxdot = dWx_dx * x1dot
        whdot = dWh_dx * x1dot + dWh_dh * x2dot
        x3_safe = max(x3, params.eps)
        x3dot = (T * np.cos(x5 + params.delta) - D) / params.m - params.g * np.sin(x4) - \
                (wxdot * np.cos(x4) + whdot * np.sin(x4))
        x4dot = (T * np.sin(x5 + params.delta)) / (params.m * x3_safe) + L / (params.m * x3_safe) - \
                params.g * np.cos(x4) / x3_safe + (wxdot * np.sin(x4) - whdot * np.cos(x4)) / x3_safe
        x5dot = u
        return np.array([x1dot, x2dot, x3dot, x4dot, x5dot])

    def rk4(x, u, t, dt, k):
        k1 = dynamics(x, u, t, k)
        k2 = dynamics(x + dt / 2 * k1, u, t + dt / 2, k)
        k3 = dynamics(x + dt / 2 * k2, u, t + dt / 2, k)
        k4 = dynamics(x + dt * k3, u, t + dt, k)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    min_hs = []
    k_samples = np.random.normal(loc=k_mid, scale=0.25, size=M)
    violating_ks = {
        'x2dotdot_low': [],
        'x2dotdot_high': []
    }

    if plot:
        plt.figure()

    for k_val in k_samples:
        xk = np.array([0, 600, 239.7, -2.249 * np.pi / 180, 7.353 * np.pi / 180])
        t = 0
        traj = [xk.copy()]
        violated_low = False
        violated_high = False

        for i in range(params.nu):
            xk = rk4(xk, u_opt[i], t, params.dut, k_val)
            traj.append(xk.copy())
            t += params.dut

            x1, x2, x3, x4, x5 = xk
            wx, wh, dWx_dx, dWh_dx, dWh_dh = wind_fields(x1, x2, k_val)
            x1dot = x3 * np.cos(x4) + wx
            x2dot = x3 * np.sin(x4) + wh
            wxdot = dWx_dx * x1dot
            whdot = dWh_dx * x1dot + dWh_dh * x2dot
            x3_safe = max(x3, params.eps)

            T = params.beta0 * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)
            D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2
            L = 0.5 * params.rho * params.S * (params.C0 + params.C1 * x5) * x3 ** 2

            x3dot = (T * np.cos(x5 + params.delta) - D) / params.m - params.g * np.sin(x4) - \
                    (wxdot * np.cos(x4) + whdot * np.sin(x4))
            x4dot = (T * np.sin(x5 + params.delta)) / (params.m * x3_safe) + L / (params.m * x3_safe) - \
                    params.g * np.cos(x4) / x3_safe + (wxdot * np.sin(x4) - whdot * np.cos(x4)) / x3_safe
            x2dotdot = x3dot * np.sin(x4) + x3 * x4dot * np.cos(x4) + whdot

            if x2dotdot < -2 * params.g:
                violated_low = True
            if x2dotdot > 10 * params.g:
                violated_high = True

        if violated_low:
            violating_ks['x2dotdot_low'].append(float(k_val))
        if violated_high:
            violating_ks['x2dotdot_high'].append(float(k_val))

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
    print('Minimum altitude over all runs: ' + str(f"{min(min_hs):.3f}"))

    M = len(min_hs)
    crash_0 = math.fsum(h <= 0 for h in min_hs) / M * 100
    fail_50 = math.fsum(h <= 50 for h in min_hs) / M * 100
    fail_100 = math.fsum(h <= 100 for h in min_hs) / M * 100

    print('Crash rate: ' + str(crash_0) + '%')
    print('Fail 50 feet rate: ' + str(fail_50) + '%')
    print('Fail 100 rate: ' + str(fail_100) + '%')

    return violating_ks




