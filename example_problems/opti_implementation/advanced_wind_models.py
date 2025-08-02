import casadi as ca

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

def analytic_wind_y(x_, h_, k_):
    h_safe = ca.fmax(h_, 10.0)
    return k_ * h_safe / h_star * B_cosh(x_)

def analytic_wind_model():
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
