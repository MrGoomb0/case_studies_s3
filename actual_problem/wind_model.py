"""
This file contains a function that returns the first wind field as described in Pesch et al. 

To create a new wind model, just follow the same skeleton as the function described below, 
as this will ensure that it works well with the other functions.
"""

from casadi import if_else, log, exp, cos, sin, MX, pi

"""
A function that given the necessary state variables and a value for 'k' 
gives the wind model as described in Pesch et al.
---
## Parameters
 - x : horizontal position state variable
 - h : vertical position state variable
 - V : velocity state variable
 - gamma : gamma state variable
 - k : the k value of the wind field

 ## Return
 - WX : horizontal wind field
 - WH : vertical wind field
 - WXdot : time derivative of the horizontal wind field
 - WHdot : time derivative of the vertical wind field
"""
def windModel(x: MX.sym, h: MX.sym, V: MX.sym, gamma: MX.sym, t: MX.sym, k : float = 1):
    # Coefficients definition
    h_star = 1000
    a = 6*1e-8
    b = -4*1e-11
    c = -log(25/30.6) * 1e-12
    d = 8.0281 * 1e-8
    e = 6.28083 * 1e-11

    # Initialisation of the piecewise functions
    A = if_else(x <= 4100,
                if_else(x <= 500, -50 + a*x**3 + b*x**4, 1/40 * (x - 2300)),
                if_else(x <= 4600, 50 - a * (4600 - x)**3 - b * (4600 - x)**4, 50)
            )
    B = if_else(x <= 4100,
                if_else(x <= 500, d*x**3 + e*x**4, -51 * exp(-c*(x - 2300)**4)),
                if_else(x <= 4600, d * (4600 - x)**3 - e * (4600 - x)**4, 0)
            )
    
    A_x = if_else(x <= 4100,
                if_else(x <= 500,  3*a*x**2 + 4*b*x**3, 1/40),
                if_else(x <= 4600, 3 * a * (4600 - x)**2 + 4 * b * (4600 - x)**3, 0)
            )
    B_x = if_else(x <= 4100,
                if_else(x <= 500, 3*d*x**2 + 4*e*x**3, 51 * 4 * (c*(x - 2300))**3 * exp(-c*(x - 2300)**4)),
                if_else(x <= 4600, - 3 * d * (4600 - x)**2 + 4 * e * (4600 - x)**3, 0)
            )
    
    # Definition of the wind variables
    WX = k * A
    WH = k * (h / h_star) * B
    
    # Definition of the partial derivatives
    WX_x = k * A_x
    WX_h = 0
    WH_x = k * h / h_star * B_x 
    WH_h = k * 1 / h_star * B

    # Definition of the wind derivatives
    WXdot = WX_x * (V * cos(gamma) + WX) + WX_h * (V * sin(gamma) + WH)
    WHdot = WH_x * (V * cos(gamma) + WX) + WH_h * (V * sin(gamma) + WH)

    return WX, WH, WXdot, WHdot


"""
A function that given the necessary state variables and a value for 'k' 
gives a very simple wind model.
---
## Parameters
 - x : horizontal position state variable
 - h : vertical position state variable
 - V : velocity state variable
 - gamma : gamma state variable
 - k : the k value of the wind field

 ## Return
 - WX : horizontal wind field
 - WH : vertical wind field
 - WXdot : time derivative of the horizontal wind field
 - WHdot : time derivative of the vertical wind field
"""
def easyWindModel(x: MX.sym, h: MX.sym, V: MX.sym, gamma: MX.sym, t: MX.sym, k: float = 1):
    WX = cos(t)
    WH = -5*t
    WXdot = -pi * sin(t * pi)
    WHdot = 5
    return WX, WH, WXdot, WHdot