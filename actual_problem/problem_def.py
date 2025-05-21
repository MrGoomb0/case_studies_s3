"""
This file contains a full description using casadi of the 2D problem as described by Pesch et al. 
"""

from casadi import *

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


"""
Function initialises all the state and control variables and return the derivative function 'f'.
---
## Parameters
 - windmodel : function returning the wind model variables and there derivatives.
 - k : wind field parameter
 - q : exponent of the Bolza objective function

## Return
 - f : derivative function
"""
def problem2D(windmodel, k=1, q=2):
    t = MX.sym('t')

    x1 = MX.sym('x1')   # x
    x2 = MX.sym('x2')   # h
    x3 = MX.sym('x3')   # V
    x4 = MX.sym('x4')   # gamma
    x5 = MX.sym('x5')   # alpha

    u = MX.sym('u')

    WX, WH, WXdot, WHdot = windmodel(x1, x2, x3, x4, t, k)

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

    return f