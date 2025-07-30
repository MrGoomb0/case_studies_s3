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

def A_piecewise(x_, s_=1):
        A1 = -50 + a * (x_/s_)**3 + b * (x_/s_)**4
        A2 = 0.025 * ((x_/s_) - 2300)
        A3 = 50 - a * (4600 - (x_/s_))**3 - b * (4600 - (x_/s_))**4
        A4 = 50
        s1 = Smooth(x_, 480 * s_, 520 * s_)
        s2 = Smooth(x_, 4080 * s_, 4120 * s_)
        s3 = Smooth(x_, 4580 * s_, 4620 * s_)
        B12 = (1 - s1)*A1 + s1*A2
        B23 = (1 - s2)*A2 + s2*A3
        B34 = (1 - s3)*A3 + s3*A4
        return ca.if_else(x_ <= 500 * s_, B12,
               ca.if_else(x_ <= 4100 * s_, B23,
               ca.if_else(x_ <= 4600 * s_, B34, A4)))


def B_piecewise(x_, s_=1):
        B1 = d * (x_/s_)**3 + e * (x_/s_)**4
        B2 = -51 * ca.exp(ca.fmin(-c * ((x_/s_) - 2300)**4, 30))
        B3 = d * (4600 - (x_/s_))**3 + e * (4600 - (x_/s_))**4
        B4 = 0
        s1 = Smooth(x_, 480 * s_, 520 * s_)
        s2 = Smooth(x_, 4080 * s_, 4120 * s_)
        s3 = Smooth(x_, 4580 * s_, 4620 * s_)
        B12 = (1 - s1)*B1 + s1*B2
        B23 = (1 - s2)*B2 + s2*B3
        B34 = (1 - s3)*B3 + s3*B4
        return ca.if_else(x_ <= 500 * s_, B12,
               ca.if_else(x_ <= 4100 * s_, B23,
               ca.if_else(x_ <= 4600 * s_, B34, B4)))

def originalWindModel(x_, h_, k_):
    return wind_x(x_, k_), wind_h(x_, h_, k_)

def wind_x(x_, k_):
    return k_ * A_piecewise(x_)

def wind_h(x_, h_, k_):
    h_safe = ca.fmax(h_, 10.0)
    return k_ * h_safe / h_star * B_piecewise(x_)

def A_cosh(x, e=1):
    m = 1/40
    a = 2000
    c = 2300 * e
    d = 150
    f = 50*d / (2*e*a) * (
        ca.log(ca.cosh( ((x-(c)) + e*a) / d)) 
        - ca.log(ca.cosh( ((x-(c)) - e*a) / d))
        )
    return f

def B_cosh(x, e=1):
    a = 1150**2
    c = 1350**2
    d = 630**2
    l = 2300 * e
    f = -25.5 + 25.5*(d*e)  /(2*e*a) * (
        ca.log(ca.cosh( ((x - l)**2 - e**2*c + e**2*a) / (d*e**2))) 
        - ca.log(ca.cosh( ((x - l)**2 - e**2*c - e**2*a) / (d*e**2))) 
        )
    return f

def analytic_wind_x(x_, k_):
    return k_ * A_cosh(x_)

def analytic_wind_h(x_, h_, k_):
    h_safe = ca.fmax(h_, 10.0)
    return k_ * h_safe / h_star * B_piecewise(x_)

def analytic_wind_model(x_, h_, k_):
    return analytic_wind_x(x_, k_), analytic_wind_h(x_, h_, k_)

def dx_A_cosh(x, e=1):
    m = 1/40
    a = 2000
    c = 2300 * e
    d = 150
    f = 50 / (2*e*a) * (
        (1 / (ca.cosh( ((x-(c)) + e*a) / d))) * ca.sinh(((x-(c)) + e*a) / d)
        - (1 / (ca.cosh( ((x-(c)) - e*a) / d))) * ca.sinh(((x-(c)) - e*a) / d)
        )
    return f

def dx_B_cosh(x, e=1):
    a = 1150**2
    c = 1350**2
    d = 630**2
    l = 2300 * e
    f = 25.5 /(2*e*a) * (
        (1/(ca.cosh( ((x - l)**2 - e*c + e*a) / (d*e)))) * ca.sinh(((x - l)**2 - e*c + e*a) / (d*e)) * 2 *(x-l)
        - (1/(ca.cosh( ((x - l)**2 - e*c - e*a) / (d*e)))) * ca.sinh(((x - l)**2 - e*c - e*a) / (d*e)) * 2 *(x-l) 
        )
    return f