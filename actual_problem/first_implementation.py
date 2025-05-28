from casadi import *
import numpy as np 

# Physical Constants
RHO = 0.2203 * 1e-2
S = 0.1560 * 1e4
G = 3.2172 * 1e1
DELTA = 2  / 180 * pi            # Has to be in rad.
MASS = 150 * 1e3

BETA0 = 0.3825
BETA0DOT = 0.2

T_0 = 0
T_SIGMA = (1 - BETA0) / BETA0DOT
T_F = 40

ALPHA_STAR = 1
ALPHA_MAX = 17.2 / 180 * pi      # Has to be in rad.

A0 = 10.4456 * 1e5
A1 = -0.2398 * 1e2
A2 = 0.1442 * 1e-1

B0 = 0.1552
B1 = 0.12369
B2 = 2.4203

C0 = 0.7125
C1 = 6.0877
C2 = -9.0277

H_R = 1000
U_MAX = 3 / 180 * pi             #  Has to be in rad.
U_MIN = -U_MAX

# Initial Conditions
X_0 = 0
GAMMA_0 = -2.249
H_0 = 600
# ALPHA_0 = 7.353 / 180 * pi
ALPHA_0 = -0.16
V_0 = 239.7
# GAMMA_F = 7.431 / 180 * pi
GAMMA_F = -1.2

def main():
    # Problem 'hyperparameters'
    N = 50
    M = 4
    q = 1

    # Problem set-up
    print("Started problem set-up.")

    t = MX.sym('t')

    x1 = MX.sym('x1')   # x
    x2 = MX.sym('x2')   # h
    x3 = MX.sym('x3')   # V
    x4 = MX.sym('x4')   # gamma
    x5 = MX.sym('x5')   # alpha

    u = MX.sym('u')

    # Wind state variables partial derivatives
    
    K = 1
    H_STAR = 1000
    
    a = 6*1e-8
    b = -4*1e-11
    c = -log(25/30.6) * 1e-12
    d = 8.0281 * 1e-8
    e = 6.28083 * 1e-11

    A = if_else(x1 <= 4100,
                if_else(x1 <= 500, -50 + a*x1**3 + b*x1**4, 1/40 * (x1 - 2300)),
                if_else(x1 <= 4600, 50 - a * (4600 - x1)**3 - b * (4600 - x1)**4, 50)
            )
    B = if_else(x1 <= 4100,
                if_else(x1 <= 500, d*x1**3 + e*x1**4, -51 * exp(-c*(x1 - 2300)**4)),
                if_else(x1 <= 4600, d * (4600 - x1)**3 - e * (4600 - x1)**4, 0)
            )
    
    A_x = if_else(x1 <= 4100,
                if_else(x1 <= 500,  3*a*x1**2 + 4*b*x1**3, 1/40),
                if_else(x1 <= 4600, 3 * a * (4600 - x1)**2 + 4 * b * (4600 - x1)**3, 0)
            )
    B_x = if_else(x1 <= 4100,
                if_else(x1 <= 500, 3*d*x1**2 + 4*e*x1**3, 51 * 4 * (c*(x1 - 2300))**3 * exp(-c*(x1 - 2300)**4)),
                if_else(x1 <= 4600, - 3 * d * (4600 - x1)**2 + 4 * e * (4600 - x1)**3, 0)
            )
    
    WX = K * A
    WH = K * (x2 / H_STAR) * B
    
    WX_x = K * A_x
    WX_h = 0
    WH_x = K * x2 / H_STAR * B_x 
    WH_h = K * 1 / H_STAR * B



    # Wind state variables derivatives
    WXdot = WX_x * (x3 * cos(x4) + WX) + WX_h * (x3 * sin(x4) + WH)
    WHdot = WH_x * (x3 * cos(x4) + WX) + WH_h * (x3 * sin(x4) + WH)

    x1dot = x3 * cos(x4) + WX
    x2dot = x3 * sin(x4) + WH

    x3dot_base = -1 / (2*MASS) * RHO  * S * (B0  + B1 * x5 * B2*x5**2) * x3**2 \
            -G * sin(x4) - (WXdot * cos(x4) + WHdot * sin(x4))
    x3dot_case_1 = x3dot_base + (A0 + A1 * x3 + A2 * x3**2) * cos(x5 + DELTA) / MASS *(BETA0 + BETA0DOT * t)     # 0 <= t <= T_SIGMA
    x3dot_case_2 = x3dot_base + (A0 + A1 * x3 + A2 * x3**2) * cos(x5 + DELTA) / MASS                             # T_SIGMA <≃ t <= T_F

    time_cond = t <= T_SIGMA

    x3dot = if_else(time_cond, x3dot_case_1, x3dot_case_2)

    x4dot_base = -G / x3 * cos(x4) + 1 / x3 * (WXdot * sin(x4) - WHdot * cos(x4))
    x4dot_case_1_1 = x4dot_base + RHO / (2*MASS) * S * x3**3 * (C0 + C1*x5) \
            + x3 * (A0 + A1*x3 + A2*x3**2) * sin(x5 + DELTA) / MASS * (BETA0 + BETA0DOT * t)  # 0 <= t <= T_SIGMA & x5 <= ALPHA_STAR
    x4dot_case_2_1 = x4dot_base + RHO / (2*MASS) * S * x3**3 * (C0 + C1*x5 + C2 * (x5 - ALPHA_STAR)**2) \
            + x3 * (A0 + A1*x3 + A2*x3**2) * sin(x5 + DELTA) / MASS * (BETA0 + BETA0DOT * t)  # 0 <= t <= T_SIGMA & ALPHA_STAR <= x5 <= ALPHA_MAX
    x4dot_case_1_2 = x4dot_base + RHO / (2*MASS) * S * x3**3 * (C0 + C1*x5) \
            + x3 * (A0 + A1*x3 + A2*x3**2) * sin(x5 + DELTA) / MASS                           # T_SIGMA <= t <= T_F & x5 <= ALPHA_STAR
    x4dot_case_2_2 = x4dot_base + RHO / (2*MASS) * S * x3**3 * (C0 + C1*x5 + C2 * (x5 - ALPHA_STAR)**2) \
            + x3 * (A0 + A1*x3 + A2*x3**2) * sin(x5 + DELTA) / MASS                          # T_SIGMA <= t <= T_F & ALPHA_STAR <= x5 <= ALPHA_MAX

    alpha_cond = x5 <= ALPHA_STAR

    x4dot = if_else(time_cond, 
                    if_else(alpha_cond, x4dot_case_1_1, x4dot_case_2_1),
                    if_else(alpha_cond, x4dot_case_1_2, x4dot_case_2_2)
                    )

    x5dot = u

    L = MX.sym('L')
    Ldot = (H_R - x1) ** q

    y = vertcat(x1, x2, x3, x4, x5, L)
    ydot = vertcat(x1dot, x2dot, x3dot, x4dot, x5dot, Ldot)

    f = Function('f', [y, u, t], [ydot])

    print("Finished problem set-up. \n")

    print("Started integrator set-up.")
    F = cranknicholson(f, T_F, N, M)
    print("Finished integrator set-up. \n")

    print("Started problem solving.")
    w_opt = nlpsolver(N, F)


def cranknicholson(f, T, N, M=1):
    h = T / N / M
    
    X0 = MX.sym('X0', 5)
    J0 = MX.sym('J0')
    Y0 = vertcat(X0, J0)
    T0 = MX.sym('T0')
    Y = MX.sym('Y', 6)
    U = MX.sym('U')
    
    Yk = Y0
    Tk = T0
    I = Function('I', [Y, U, Tk, Yk], [Y - Yk - h / 2 * (f(Y, U, Tk) + f(Yk, U, Tk + h))], ['y0', 'p', 'tk', 'yk'], ['If'])

    G = rootfinder('G', 'fast_newton', I)
    
    Y_guess = Yk
    for j in range(M):
        Y_guess = G(Y_guess, U, Tk, Yk)
        Yk = Y_guess
        Tk = Tk + h
    Xk = Y_guess[0:5]
    Jk = Y_guess[5]

    F = Function('F', [X0, J0, T0, U], [Xk, Jk, Tk], ['x0', 'j0', 't0', 'p'], ['xf', 'jf', 'tf'])
    return F

def nlpsolver(N, F):
    Tk = T_0
    w = []
    w0 = []
    lbw, ubw = [], []
    Jk = 0
    g = []
    lbg, ubg = [], []

    Xk = MX.sym('X0', 5)

    w += [Xk]
    # lbw += [X_0, H_0, V_0, GAMMA_0, -inf]
    # ubw += [X_0, H_0, V_0, GAMMA_0, ALPHA_MAX]
    # w0 += [X_0, H_0, V_0, GAMMA_0, 0]
    lbw += [-inf, -inf, -inf, -inf, -inf]
    ubw += [inf, inf, inf, inf, inf]
    w0 += [X_0, H_0, V_0, GAMMA_0, 0]

    for k in range(N):
        Uk = MX.sym('U_' + str(k))
        w += [Uk]
        lbw += [U_MIN]
        ubw += [U_MAX]
        w0 += [U_MAX]

        Fk = F(x0=Xk, j0=Jk, t0=Tk, p=Uk)
        Tk = Fk['tf']
        Xk_end = Fk['xf']
        Jk = Fk['jf']

        Xk = MX.sym('X_' + str(k+1), 5)
        w += [Xk]
        if k == N - 1:
            lbw += [-inf, -inf, -inf, -inf, -inf]
            ubw += [inf, inf, inf, inf, ALPHA_MAX]
        else:
            lbw += [-inf, -inf, -inf, -inf, -inf]
            ubw += [inf, inf, inf, inf, ALPHA_MAX]
        w0 += [X_0, H_0, V_0, GAMMA_0, 0]
        # w0 += [0, H_0, 0, 0, 0]


        g += [Xk_end - Xk]
        lbg += [0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0]
    J = Jk
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', prob)
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x'].full().flatten()
    return w_opt

if __name__ == "__main__":
    main()

