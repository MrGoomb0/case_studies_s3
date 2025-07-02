"""
Possible wind model functions and approximations used in the advanced multiple planes solver.
"""

import casadi as ca
import numpy as np

# Wind model x parameters (piecewise smooth wind)
a = 6e-8  # x transition midpoint [ft]
b = -4e-11  # second transition point [ft]

# Wind model h parameters (polynomial form)
c = -np.log(25 / 30.6) * 1e-12  # transition smoothing width [ft]
d = -8.02881e-8  # polynomial coeff [sec^-1 ft^-2]
e = 6.28083e-11  # polynomial coeff [sec^-1 ft^-3]
h_star = 1000  # used in some wind models

eps = 1e-6  # to avoid division by zero in V

def Smooth(x_, x0, x1):
    t = (x_ - x0) / (x1 - x0 + eps)
    return ca.if_else(
        x_ < x0, 0, ca.if_else(x_ > x1, 1, 6 * t**5 - 15 * t**4 + 10 * t**3)
    )

def A_piecewise(x_):
    A1 = -50 + a * x_**3 + b * x_**4
    A2 = 0.025 * (x_ - 2300)
    A3 = 50 - a * (4600 - x_) ** 3 - b * (4600 - x_) ** 4
    A4 = 50
    s1 = Smooth(x_, 480, 520)
    s2 = Smooth(x_, 4080, 4120)
    s3 = Smooth(x_, 4580, 4620)
    B12 = (1 - s1) * A1 + s1 * A2
    B23 = (1 - s2) * A2 + s2 * A3
    B34 = (1 - s3) * A3 + s3 * A4
    return ca.if_else(
        x_ <= 500, B12, ca.if_else(x_ <= 4100, B23, ca.if_else(x_ <= 4600, B34, A4))
    )


def B_piecewise(x_):
    B1 = d * x_**3 + e * x_**4
    B2 = -51 * ca.exp(ca.fmin(-c * (x_ - 2300) ** 4, 30))
    B3 = d * (4600 - x_) ** 3 + e * (4600 - x_) ** 4
    B4 = 0
    s1 = Smooth(x_, 480, 520)
    s2 = Smooth(x_, 4080, 4120)
    s3 = Smooth(x_, 4580, 4620)
    B12 = (1 - s1) * B1 + s1 * B2
    B23 = (1 - s2) * B2 + s2 * B3
    B34 = (1 - s3) * B3 + s3 * B4
    return ca.if_else(
        x_ <= 500, B12, ca.if_else(x_ <= 4100, B23, ca.if_else(x_ <= 4600, B34, B4))
    )

def originalWindModel(x_, h_, k_):
    return wind_x(x_, k_), wind_h(x_, h_, k_)

def wind_x(x_, k_):
    return k_ * A_piecewise(x_)

def wind_h(x_, h_, k_):
    h_safe = ca.fmax(h_, 10.0)
    return k_ * h_safe / h_star * B_piecewise(x_)

