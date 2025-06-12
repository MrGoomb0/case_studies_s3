import math

from casadi import *
from dataclasses import dataclass
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed


@dataclass
class Parameters:
    # Time and discretization
    tf: float = 50          # final time [sec]
    nu: int = 100           # number of control intervals
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
    eps: float = 1e-6  # to avoid division by zero in V


params = Parameters()


def solve_ocp_multi_plane_min_h(k_values: list[float]):
    '''
    Input:
        k_values (list with floats): The K values for the different planes
    Return:
        w_opt (list with floats): Optimal decision variables (feed to multi_plane_plot_2D)
                                    [h_min, controls, state for plane 1,..., states for plane n_k]
        J_opt (float): optimal objective value
    '''

    nk = len(k_values) # Number of k:s

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

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2

    # nlp prep
    w = []  # List of decision variables
    w0 = []  # Initial guess
    lbw = []  # Lower bounds
    ubw = []  # Upper bounds
    g = []  # Constraints
    lbg = []  # Lower bound on constraints
    ubg = []  # Upper bound on constraints

    # control
    U = MX.sym('U', params.nu)
    w += [U]
    lbw += [-3 * pi / 180]*params.nu
    ubw += [3 * pi / 180]*params.nu
    w0 += [0]*params.nu

    # minimal height, (to be the objective, min -min_h)
    min_h = MX.sym('min_h')
    w += [min_h]
    lbw += [0]
    ubw += [inf]
    w0 += [300]

    # Add each plane to nlp definition
    for i in range(nk):
        k_value = k_values[i]

        # wind
        wind_x_expr = k_value * A
        wind_x = Function('wind_x', [x1], [wind_x_expr])
        wind_h_expr = k_value * x2 * B / params.h_star
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

        f = Function('f', [x, u, t], [vertcat(x1dot, x2dot, x3dot, x4dot, x5dot)])

        # For h¨(t) constraint
        x2dotdot_expr = x3dot * sin(x4) + x3 * x4dot * cos(x4) + whdot
        x2dotdot = Function('x2dotdot', [x, u, t], [x2dotdot_expr])

        # Integration
        def rk4_step(xk, uk, tk, dt):
            k1 = f(xk, uk, tk)
            k2 = f(xk + dt / 2 * k1, uk, tk + dt / 2)
            k3 = f(xk + dt / 2 * k2, uk, tk + dt / 2)
            k4 = f(xk + dt * k3, uk, tk + dt)
            return xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Initial condition
        x_initial = [0, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
        Xk = MX.sym('X_0'+ str(i), 5)
        w += [Xk]
        lbw += x_initial
        ubw += x_initial
        w0 += x_initial

        Tk = 0

        # nlp set up
        for k in range(params.nu):
            Uk = U[k]
            # state integration
            Xk_end = rk4_step(Xk, Uk, Tk, params.dut)

            h_ddot_val = x2dotdot(Xk, Uk, Tk)
            Tk += params.dut

            # add state to nlp
            Xk = MX.sym('X_' + str(k + 1) + str(i), 5)
            w += [Xk]
            lbw += [-inf, 0, -inf, -inf, -inf]
            ubw += [inf, inf, inf, inf, 17.2 * pi / 180]
            w0 += [k * 10000 / params.nu, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
            g += [Xk_end - Xk]
            lbg += [0, 0, 0, 0, 0]
            ubg += [0, 0, 0, 0, 0]

            # Constraints
            g += [Xk[1] - min_h]
            lbg += [0]
            ubg += [inf]
            g += [h_ddot_val]
            lbg += [-0.5 * params.g]
            ubg += [1.5 * params.g]

        # Finial state constraint
        g += [Xk[3]]
        lbg += [0]
        ubg += [inf]
    # Objective function
    J = -min_h

    opts = {'ipopt': {'print_level': 3}}
    nlp = {'x': vertcat(*w), 'f': J, 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', nlp, opts)
    arg = {'x0': w0, 'lbx': lbw, 'ubx': ubw, 'lbg': lbg, 'ubg': ubg}

    # Solve
    sol = solver(**arg)
    w_opt = sol['x'].full().flatten()
    J_opt = sol['f'].full().flatten()

    return w_opt, J_opt


def solve_ocp_multi_plane_min_h_initial_geuss_help(k_values: list[float], state_guess: list[float], control_guess: list[float]):
    '''
    Input:
        k_values (list with floats): The K values for the different planes
        state_guess (list with floats): A initial guess for the decision variable corresponding to the state
        control_guess (list with floats): A initial guess for the decision variable corresponding to the control
    Return:
        w_opt (list with floats): Optimal decision variables (feed to multi_plane_plot_2D)
                                    [h_min, controls, state for plane 1,..., states for plane n_k]
        J_opt (float): optimal objective value
    '''
    nk = len(k_values)

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

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2

    # nlp prep
    w = []  # List of decision variables
    w0 = []  # Initial guess
    lbw = []  # Lower bounds
    ubw = []  # Upper bounds
    g = []  # Constraints
    lbg = []  # Lower bound on constraints
    ubg = []  # Upper bound on constraints

    # control
    U = MX.sym('U', params.nu)
    w += [U]
    lbw += [-3 * pi / 180]*params.nu
    ubw += [3 * pi / 180]*params.nu
    w0 += control_guess

    min_h = MX.sym('min_h')
    w += [min_h]
    lbw += [0]
    ubw += [inf]
    w0 += [300]

    for i in range(nk):
        k_value = k_values[i]

        wind_x_expr = k_value * A
        wind_x = Function('wind_x', [x1], [wind_x_expr])
        wind_h_expr = k_value * x2 * B / params.h_star
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


        # initial condition
        x_initial = [0, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
        Xk = MX.sym('X_0'+ str(i), 5)
        w += [Xk]
        w0 += state_guess[0:5]
        lbw += x_initial
        ubw += x_initial

        Tk = 0

        # nlp set up
        for k in range(params.nu):
            Uk = U[k]
            # state integration
            Xk_end = rk4_step(Xk, Uk, Tk, params.dut)
            h_ddot_val = x2dotdot(Xk, Uk, Tk)

            Tk += params.dut

            # state
            Xk = MX.sym('X_' + str(k + 1) + str(i), 5)
            w += [Xk]
            start_idx = 5 * (k + 1)
            end_idx = start_idx + 5
            w0 += state_guess[start_idx:end_idx]
            lbw += [-inf, 0, -inf, -inf, -inf]
            ubw += [inf, inf, inf, inf, 17.2 * pi / 180]
            g += [Xk_end - Xk]
            lbg += [0, 0, 0, 0, 0]
            ubg += [0, 0, 0, 0, 0]

            g += [Xk[1] - min_h]
            lbg += [0]
            ubg += [inf]

            # Add to constraint list
            g += [h_ddot_val]
            lbg += [-0.5 * params.g]
            ubg += [1.5 * params.g]

        # finial state constraint
        g += [Xk[3]]
        lbg += [0]
        ubg += [inf]

    J = -min_h  # objective function

    opts = {'ipopt': {'print_level': 3 }}
    nlp = {'x': vertcat(*w), 'f': J, 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', nlp, opts)

    arg = {'x0': w0, 'lbx': lbw, 'ubx': ubw, 'lbg': lbg, 'ubg': ubg}

    # Solve
    sol = solver(**arg)
    w_opt = sol['x'].full().flatten()
    J_opt = sol['f'].full().flatten()

    return w_opt, J_opt


def u_opt_return(w_opt: list[float]):
    # returns the optimal control from the result of the solver
    u_opt = w_opt[:params.nu]
    return u_opt


def plot_multi_plane_2D(w_opt: list[float], k_values: list[float]):
    # plots the optimal trajectories

    # Prep
    N = params.nu
    tf = params.tf
    nk = len(k_values)
    u_opt = w_opt[:N]
    h_min = w_opt[N]
    w = w_opt[N + 1:]
    # going over all k values
    all_lists = []
    for i in range(nk):
        new_w = w[:N * 5 + 5]
        all_lists.append(new_w)
        w = w[N * 5 + 5:]

    # plot
    tgrid = [tf / N * k for k in range(N + 1)]
    plt.figure(1)
    plt.clf()
    for i in range(nk):
        x1_opt = all_lists[i][0::5]
        x2_opt = all_lists[i][1::5]
        x3_opt = all_lists[i][2::5]
        x4_opt = all_lists[i][3::5]
        x5_opt = all_lists[i][4::5]

        print(min(x2_opt))

        plt.plot(x1_opt, x2_opt, '--', label=f'k = {k_values[i]:.3f}')
    plt.grid()
    plt.xlabel('horizontal distance [ft]')
    plt.ylabel('altitude [ft]')
    plt.legend()
    plt.show()

    plt.figure(nk + 1)
    plt.clf()
    plt.step(tgrid[:-1], u_opt, where='post')
    plt.xlabel('t')
    plt.title('Optimal Control')
    plt.grid()
    plt.show()

    return


def reconstruction_multi_plane_2D(u_opt: list[float], k_value: float, multiplier: int = 1):
    '''
    Input:
        u_opt (list with floats): The computed optimal control from one of the solvers
        k_value (float): K value for wind corresponding to the trajectory that is to computed
        multiplier (int): If the trajactory needs to be computed with a smaller time step then the control
    Return:
        sol: A dictionary containing the following keys:
            - 'x' (list[float]): horizontal distance
            - 'h' (list[float]): altidude
            - 'V' (list[float]): speed
            - 'gamma' (list[float]): gamma
            - 'alpha' (list[float]): angle of attack alpha
            - 'u' (list[float]): control input
            - 't_grid' (list[float]): Time grid values
    '''

    N = params.nu * multiplier
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

    x2dotdot = x3dot * np.sin(x4) + x3 * x4dot * np.cos(x4) + whdot

    f = Function('f', [x, u, t], [vertcat(x1dot, x2dot, x3dot, x4dot, x5dot)])
    fun_hdotdot = Function('fun_hdotodt', [x, u, t], [x2dotdot])

    # Integration
    def rk4_step(xk, uk, tk, dt):
        k1 = f(xk, uk, tk)
        k2 = f(xk + dt / 2 * k1, uk, tk + dt / 2)
        k3 = f(xk + dt / 2 * k2, uk, tk + dt / 2)
        k4 = f(xk + dt * k3, uk, tk + dt)
        return xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # Prep
    U = []
    for k in range(params.nu):
        for i in range(multiplier):
            U.append(u_opt[k])

    X = []

    # initial condition
    Xk = [0, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
    X.extend(Xk)
    Tk = 0

    H_dotdot = []

    for k in range(N):
        H_dotdot.append(fun_hdotdot(Xk, U[k], Tk).full().flatten())
        Xk = rk4_step(Xk, U[k], Tk, dxt)
        Tk += dxt
        X.extend(Xk.full().flatten())

    T = [params.tf / N * k for k in range(N + 1)]

    return {'x': X[0::5],'h': X[1::5],'V': X[2::5],'gamma': X[3::5],'alpha': X[4::5],'u':U,'t_grid':T, 'hdotdot': H_dotdot}

def simulate_trajectories(u_opt: list[float], M: int, k_mid: float, plot=True):
    '''
    Input:
        u_opt (list with floats): The computed optimal control
        M (int): Number of MC samples
    Return:
        dict: Lists of k-values violating the x2dotdot constraints

    This is the MC simulation, it plots all simulated trajectories with k~N(1,0.25)
    '''
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
        violated_alpha = False

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

            if x2dotdot < -0.5 * params.g:
                violated_low = True
            if x2dotdot > 1.5 * params.g:
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


'''
The last two solvers can be ignored

The two solvers bellow are attempts at using smooth wind,
as well as an attempted to optimize code by making the k_value symbolic
'''

def solve_ocp_multi_plane_min_h_smooth(k_values):
    nk = len(k_values)

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
    def Smooth(x_start, x_end):
        t_smooth = (x1 - x_start) / (x_end - x_start + params.eps)
        return if_else(x1 < x_start, 0, if_else(x1 > x_end, 1, 6 * t_smooth ** 5 - 15 * t_smooth ** 4 + 10 * t_smooth ** 3))

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

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2

    # nlp prep
    w = []  # List of decision variables
    w0 = []  # Initial guess
    lbw = []  # Lower bounds
    ubw = []  # Upper bounds
    g = []  # Constraints
    lbg = []  # Lower bound on constraints
    ubg = []  # Upper bound on constraints

    # control
    U = MX.sym('U', params.nu)
    w += [U]
    lbw += [-3 * pi / 180]*params.nu
    ubw += [3 * pi / 180]*params.nu
    w0 += [0]*params.nu

    min_h = MX.sym('min_h')
    w += [min_h]
    lbw += [0]
    ubw += [inf]
    w0 += [300]

    for i in range(nk):
        k_value = k_values[i]

        wind_x_expr = k_value * A
        wind_x = Function('wind_x', [x1], [wind_x_expr])
        wind_h_expr = k_value * x2 * B / params.h_star
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


        # initial condition
        x_initial = [0, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
        Xk = MX.sym('X_0'+ str(i), 5)
        w += [Xk]
        lbw += x_initial
        ubw += x_initial
        w0 += x_initial

        Tk = 0

        # nlp set up
        for k in range(params.nu):
            Uk = U[k]
            # state integration
            Xk_end = rk4_step(Xk, Uk, Tk, params.dut)
            h_ddot_val = x2dotdot(Xk, Uk, Tk)

            Tk += params.dut

            # state
            Xk = MX.sym('X_' + str(k + 1) + str(i), 5)
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

        # finial state constraint
        g += [Xk[3]]
        lbg += [0]
        ubg += [inf]

    J = -min_h  # objective function

    opts = {'ipopt': {'print_level': 3}}

    nlp = {'x': vertcat(*w), 'f': J, 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', nlp, opts)
    arg = {'x0': w0, 'lbx': lbw, 'ubx': ubw, 'lbg': lbg, 'ubg': ubg}

    # Solve
    sol = solver(**arg)
    w_opt = sol['x'].full().flatten()
    J_opt = sol['f'].full().flatten()

    return w_opt, J_opt


def solve_ocp_multi_plane_min_h_k_sym(k_values):
    nk = len(k_values)

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

    # Other functions
    C_L = if_else(x5 > params.alpha_star, params.C0 + params.C1 * x5,
                  params.C0 + params.C1 * x5 + params.C2 * (x5 - params.alpha_star) ** 2)

    beta = if_else(t < params.sigma,
                   params.beta0 + params.beta_dot0 * t, 1.0)

    T = beta * (params.A0 + params.A1 * x3 + params.A2 * x3 ** 2)

    D = 0.5 * (params.B0 + params.B1 * x5 + params.B2 * x5 ** 2) * params.rho * params.S * x3 ** 2

    L = 0.5 * params.rho * params.S * C_L * x3 ** 2

    # nlp prep
    w = []  # List of decision variables
    w0 = []  # Initial guess
    lbw = []  # Lower bounds
    ubw = []  # Upper bounds
    g = []  # Constraints
    lbg = []  # Lower bound on constraints
    ubg = []  # Upper bound on constraints

    # control
    U = MX.sym('U', params.nu)
    w += [U]
    lbw += [-3 * pi / 180]*params.nu
    ubw += [3 * pi / 180]*params.nu
    w0 += [0]*params.nu

    min_h = MX.sym('min_h')
    w += [min_h]
    lbw += [0]
    ubw += [inf]
    w0 += [300]

    k_sym = SX.sym("k")

    wind_x_expr = k_sym * A
    wind_x = Function('wind_x', [x1,k_sym], [wind_x_expr])
    wind_h_expr = k_sym * x2 * B / params.h_star
    wind_h = Function('wind_h', [x1, x2,k_sym], [wind_h_expr])

    dWx_dx = Function("dWx_dx", [x1,k_sym], [gradient(wind_x_expr, x1)])
    dWh_dx = Function("dWh_dx", [x1, x2,k_sym], [gradient(wind_h_expr, x1)])
    dWh_dh = Function("dWh_dh", [x1, x2,k_sym], [gradient(wind_h_expr, x2)])

    for i in range(nk):
        k_value = k_values[i]
        wx = wind_x(x1, k_value)
        wh = wind_h(x1, x2, k_value)

        # ode
        x1dot = x3 * cos(x4) + wx
        x2dot = x3 * sin(x4) + wh

        wxdot = dWx_dx(x1, k_value) * x1dot
        whdot = dWh_dx(x1, x2, k_value) * x1dot + dWh_dh(x1, x2, k_value) * x2dot

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


        # initial condition
        x_initial = [0, 600, 239.7, -2.249 * pi / 180, 7.353 * pi / 180]
        Xk = MX.sym('X_0'+ str(i), 5)
        w += [Xk]
        lbw += x_initial
        ubw += x_initial
        w0 += x_initial

        Tk = 0

        # nlp set up
        for k in range(params.nu):
            Uk = U[k]
            # state integration
            Xk_end = rk4_step(Xk, Uk, Tk, params.dut)
            h_ddot_val = x2dotdot(Xk, Uk, Tk)

            Tk += params.dut

            # state
            Xk = MX.sym('X_' + str(k + 1) + str(i), 5)
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

        # finial state constraint
        g += [Xk[3]]
        lbg += [0]
        ubg += [inf]

    J = -min_h  # objective function

    opts = {'ipopt': { 'print_level': 3}}
    nlp = {'x': vertcat(*w), 'f': J, 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', nlp, opts)
    arg = {'x0': w0, 'lbx': lbw, 'ubx': ubw, 'lbg': lbg, 'ubg': ubg}

    # Solve
    sol = solver(**arg)
    w_opt = sol['x'].full().flatten()
    J_opt = sol['f'].full().flatten()

    return w_opt, J_opt




