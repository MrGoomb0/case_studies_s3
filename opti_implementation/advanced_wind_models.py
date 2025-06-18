import casadi as ca

def A_cosh(x, e=1, central_point=2300):
    m = 1/40
    a = 2000
    c = central_point
    d = 150
    f = 50*d / (2*e*a) * (
        ca.log(ca.cosh( ((x-(c)) + e*a) / d)) 
        - ca.log(ca.cosh( ((x-(c)) - e*a) / d))
        )
    return f

def B_cosh(x, e=1, central_point=2300):
    a = 1150**2
    c = 1350**2
    d = 630**2
    l = central_point
    f = -25.5 + 25.5*(d*e)  /(2*e*a) * (
        ca.log(ca.cosh( ((x - l)**2 - e*c + e*a) / (d*e))) 
        - ca.log(ca.cosh( ((x - l)**2 - e*c - e*a) / (d*e))) 
        )
    return f


