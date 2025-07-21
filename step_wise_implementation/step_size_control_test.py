from step_size_control import refineSolution, reduceIntervalBy2, rk45, stepSizedIntegrationRefinement
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from wind_models import A_piecewise, B_piecewise, dx_A_cosh, dx_B_cosh, A_cosh, B_cosh

TEST_1 = False
TEST_2 = False
TEST_3 = False
TEST_4 = False
TEST_5 = False
TEST_6 = False
TEST_7 = False
TEST_8 = False
TEST_10 = False
TEST_11 = False
TEST_12 = False
TEST_13 = False
TEST_14 = False

def main():
    if TEST_1:
        test_1()
    if TEST_2:
        test_2()
    if TEST_3:
        test_3()
    if TEST_4:
        test_4()
    if TEST_5:
        test_5()
    if TEST_6:
        test_6()
    if TEST_7:
        test_7()
    if TEST_8:
        test_8()
    if TEST_10:
        test_10()
    if TEST_11:
        test_11()
    if TEST_12:
        test_12()
    if TEST_13:
        test_13()
    if TEST_14:
        test_14()

"""
Integration with repeated discretised step-size control for A_cosh
"""
def test_14():
    u_final = 4600
    N = 80
    tol = 10**-5
    x = ca.MX.sym('x')
    t = ca.MX.sym('t')
    u = ca.MX.sym('u')
    xdot = dx_A_cosh(u)
    f = ca.Function('f', [x, u, t], [xdot])
    uk = np.linspace(0, u_final, N-1)
    ts = np.linspace(0, u_final, N)
    dt = u_final / N
    xk = -49.963
    for i in range(2, 5):
        X, T = stepSizedIntegrationRefinement(f, xk, ts, uk, dt_min=dt/2**i, dt_max=dt*i, tol=tol)
        ts = T
        uk = T[0:-1]
        print(len(T))
    fig, axs = plt.subplots(3, 1)
    u_grid = np.linspace(0, u_final, 200)
    axs[0].plot(u_grid, A_original(u_grid))
    axs[0].scatter(T, X)
    axs[1].scatter(np.linspace(0, 1, N), np.zeros(N))
    axs[2].scatter(T, np.zeros(T.shape))
    axs[1].set_yticks([])    
    axs[2].set_yticks([])
    plt.show()

"""
Integration with discretised step-size control for A_cosh
"""
def test_13():
    u_final = 4600
    N = 80
    tol = 10**-4
    x = ca.MX.sym('x')
    t = ca.MX.sym('t')
    u = ca.MX.sym('u')
    xdot = dx_A_cosh(u)
    f = ca.Function('f', [x, u, t], [xdot])
    uk = np.linspace(0, u_final, N-1)
    ts = np.linspace(0, u_final, N)
    dt = u_final / N
    xk = -49.963
    X, T = stepSizedIntegrationRefinement(f, xk, ts, uk, dt_min=dt/2, dt_max=dt*2, tol=tol)
    print(len(T))
    fig, axs = plt.subplots(3, 1)
    u_grid = np.linspace(0, u_final, 200)
    axs[0].plot(u_grid, A_original(u_grid))
    axs[0].scatter(T, X)
    axs[1].scatter(np.linspace(0, 1, N), np.zeros(N))
    axs[2].scatter(T, np.zeros(T.shape))
    axs[1].set_yticks([])    
    axs[2].set_yticks([])
    plt.show()

"""
Integration with step-size control for B_original (smoothed)
"""
def test_12():
    t_final = 4600
    N = 80
    tol = 10**-4
    x = ca.MX.sym('x')
    t = ca.MX.sym('t')
    u = ca.MX.sym('u')
    xdot = ca.gradient(B_original(t), t)
    f = ca.Function('f', [x, u, t], [xdot])
    uk = 0 # Dummy value
    tk = 0
    dt = t_final / N
    xk = 0
    T = [tk]
    X = [xk]
    while tk < t_final:
        xk, dt = rk45(f, xk, uk, tk, dt, dt_min=10**-4, dt_max=150, tol=tol, t_final=t_final)
        X.append(float(xk))
        tk += dt
        T.append(float(tk))
    print(len(T))
    T = np.array(T)
    X = np.array(X)
    fig, axs = plt.subplots(3, 1)
    t_grid = np.linspace(0, t_final, 200)
    axs[0].plot(t_grid, B_original(t_grid))
    axs[0].scatter(T, X)
    axs[1].scatter(np.linspace(0, 1, N), np.zeros(N))
    axs[2].scatter(T, np.zeros(T.shape))
    axs[1].set_yticks([])    
    axs[2].set_yticks([])
    plt.show()


"""
Integration with step-size control for A_original (smoothed)
"""
def test_11():
    t_final = 4600
    N = 80
    tol = 10**-4
    x = ca.MX.sym('x')
    t = ca.MX.sym('t')
    u = ca.MX.sym('u')
    xdot = ca.gradient(A_original(t), t)
    f = ca.Function('f', [x, u, t], [xdot])
    uk = 0 # Dummy value
    tk = 0
    dt = t_final / N
    xk = -50
    T = [tk]
    X = [xk]
    while tk < t_final:
        xk, dt = rk45(f, xk, uk, tk, dt, dt_min=10**-4, dt_max=150, tol=tol, t_final=t_final)
        X.append(float(xk))
        tk += dt
        T.append(float(tk))
    print(len(T))
    T = np.array(T)
    X = np.array(X)
    fig, axs = plt.subplots(3, 1)
    t_grid = np.linspace(0, t_final, 200)
    axs[0].plot(t_grid, A_original(t_grid))
    axs[0].scatter(T, X)
    axs[1].scatter(np.linspace(0, 1, N), np.zeros(N))
    axs[2].scatter(T, np.zeros(T.shape))
    axs[1].set_yticks([])    
    axs[2].set_yticks([])
    plt.show()

"""
Integration with step-size control for A_piecewise (smoothed)
"""
def test_10():
    t_final = 4600
    N = 80
    tol = 10**-4
    x = ca.MX.sym('x')
    t = ca.MX.sym('t')
    u = ca.MX.sym('u')
    xdot = ca.gradient(B_piecewise(t), t)
    f = ca.Function('f', [x, u, t], [xdot])
    uk = 0 # Dummy value
    tk = 0
    dt = t_final / N
    xk = 0
    T = [tk]
    X = [xk]
    while tk < t_final:
        xk, dt = rk45(f, xk, uk, tk, dt, dt_min=10**-4, dt_max=150, tol=tol, t_final=t_final)
        X.append(float(xk))
        tk += dt
        T.append(float(tk))
    print(len(T))
    T = np.array(T)
    X = np.array(X)
    fig, axs = plt.subplots(3, 1)
    t_grid = np.linspace(0, t_final, 200)
    axs[0].plot(t_grid, B_piecewise(t_grid))
    axs[0].scatter(T, X)
    axs[1].scatter(np.linspace(0, 1, N), np.zeros(N))
    axs[2].scatter(T, np.zeros(T.shape))
    axs[1].set_yticks([])    
    axs[2].set_yticks([])
    plt.show()

"""
Integration with step-size control for A_piecewise (smoothed)
"""
def test_9():
    t_final = 4600
    N = 80
    tol = 10**-4
    x = ca.MX.sym('x')
    t = ca.MX.sym('t')
    u = ca.MX.sym('u')
    xdot = ca.gradient(A_piecewise(t), t)
    f = ca.Function('f', [x, u, t], [xdot])
    uk = 0 # Dummy value
    tk = 0
    dt = t_final / N
    xk = -50
    T = [tk]
    X = [xk]
    while tk < t_final:
        xk, dt = rk45(f, xk, uk, tk, dt, dt_min=10**-4, dt_max=150, tol=tol, t_final=t_final)
        X.append(float(xk))
        tk += dt
        T.append(float(tk))
    print(len(T))
    T = np.array(T)
    X = np.array(X)
    fig, axs = plt.subplots(3, 1)
    t_grid = np.linspace(0, t_final, 200)
    axs[0].plot(t_grid, A_piecewise(t_grid))
    axs[0].scatter(T, X)
    axs[1].scatter(np.linspace(0, 1, N), np.zeros(N))
    axs[2].scatter(T, np.zeros(T.shape))
    axs[1].set_yticks([])    
    axs[2].set_yticks([])
    plt.show()

"""
Integration with step-size control for B_cosh
"""
def test_8():
    t_final = 4600
    N = 80
    tol = 10**-4
    x = ca.MX.sym('x')
    t = ca.MX.sym('t')
    u = ca.MX.sym('u')
    xdot = dx_B_cosh(t)
    f = ca.Function('f', [x, u, t], [xdot])
    uk = 0 # Dummy value
    tk = 0
    dt = t_final / N
    xk = 0
    T = [tk]
    X = [xk]
    while tk < t_final:
        xk, dt = rk45(f, xk, uk, tk, dt, dt_min=10**-4, dt_max=150, tol=tol, t_final=t_final)
        X.append(float(xk))
        tk += dt
        T.append(float(tk))
    print(len(T))
    T = np.array(T)
    X = np.array(X)
    fig, axs = plt.subplots(3, 1)
    t_grid = np.linspace(0, t_final, 200)
    axs[0].plot(t_grid, B_cosh(t_grid))
    axs[0].scatter(T, X)
    axs[1].scatter(np.linspace(0, 1, N), np.zeros(N))
    axs[2].scatter(T, np.zeros(T.shape))
    axs[1].set_yticks([])    
    axs[2].set_yticks([])
    plt.show()

"""
Integration with step-size control for A_cosh
"""
def test_7():
    t_final = 4600
    N = 80
    tol = 10**-4
    x = ca.MX.sym('x')
    t = ca.MX.sym('t')
    u = ca.MX.sym('u')
    xdot = dx_A_cosh(t)
    f = ca.Function('f', [x, u, t], [xdot])
    uk = 0 # Dummy value
    tk = 0
    dt = t_final / N
    xk = -49.963
    T = [tk]
    X = [xk]
    while tk < t_final:
        xk, dt = rk45(f, xk, uk, tk, dt, dt_min=10**-4, dt_max=150, tol=tol, t_final=t_final)
        X.append(float(xk))
        tk += dt
        T.append(float(tk))
    print(len(T))
    T = np.array(T)
    X = np.array(X)
    fig, axs = plt.subplots(3, 1)
    t_grid = np.linspace(0, t_final, 200)
    axs[0].plot(t_grid, A_cosh(t_grid))
    axs[0].scatter(T, X)
    axs[1].scatter(np.linspace(0, 1, N), np.zeros(N))
    axs[2].scatter(T, np.zeros(T.shape))
    axs[1].set_yticks([])    
    axs[2].set_yticks([])
    plt.show()

"""
Discretisation refinement for B_original
"""
def test_6():
    N = 3
    t_final = 4600
    max_iter = 10
    N = 3
    tol = 10**-5
    t1 = np.linspace(0, t_final, N)
    t2 = np.linspace(0, t_final, 2*N - 1)

    x = ca.MX.sym('x')
    t = ca.MX.sym('t')

    xdot = ca.gradient(B_original(t), t)
    f = ca.Function('f', [x, t], [xdot])
    
    for iteration in range(max_iter):
        print("iteration ", iteration + 1)
        x1 = np.zeros(t1.shape)
        for i in range(t1.shape[-1]-1):
            x1[i + 1] = rk4(f, xk=x1[i], tk=t1[i], dt=(t1[i + 1] - t1[i]))

        x2 = np.zeros(t2.shape) 
        for i in range(t2.shape[-1] - 1):
            x2[i + 1] = rk4(f, x2[i], t2[i], t2[i + 1] - t2[i])

        sol1 = {'x': x1, 'h': np.zeros(x1.shape), 'V': np.zeros(x1.shape), 'gamma': np.zeros(x1.shape), 'alpha': np.zeros(x1.shape)}
        sol2 = {'x': x2, 'h': np.zeros(x2.shape), 'V': np.zeros(x2.shape), 'gamma': np.zeros(x2.shape), 'alpha': np.zeros(x2.shape)}

        t_refined = refineSolution(sol1, sol2, t1, t2, tol=tol, p=4)
        if t_refined.shape == t1.shape:
            print("Solution found after", iteration, " iterations.") 
            break
        else:
            t1 = t_refined
            t2 = reduceIntervalBy2(t1)

    x1 = np.zeros(t1.shape)
    for i in range(t1.shape[-1] -1):
        x1[i + 1] = rk4(f, xk=x1[i], tk=t1[i], dt=(t1[i + 1] - t1[i]))

    fig, axs = plt.subplots(3, 1)
    t_grid = np.linspace(0, t_final, 200)
    axs[0].plot(t_grid, B_original(t_grid))
    axs[0].scatter(t_refined, x1)
    axs[1].scatter(np.linspace(0, 1, N), np.zeros(N))
    axs[2].scatter(t_refined, np.zeros(t_refined.shape))
    axs[1].set_yticks([])    
    axs[2].set_yticks([])
    plt.show()

"""
Discretisation refinement for A_original
"""
def test_5():
    t_final = 4600
    max_iter = 10
    N = 2
    tol = 10**-5
    t1 = np.linspace(0, t_final, N)
    t2 = np.linspace(0, t_final, 2*N - 1)

    x = ca.MX.sym('x')
    t = ca.MX.sym('t')

    xdot = ca.gradient(A_original(t), t)
    f = ca.Function('f', [x, t], [xdot])
    
    for iteration in range(max_iter):
        print("iteration ", iteration + 1)
        x1 = np.ones(t1.shape) * -49.963
        for i in range(t1.shape[-1]-1):
            x1[i + 1] = rk4(f, xk=x1[i], tk=t1[i], dt=(t1[i + 1] - t1[i]))

        x2 = np.ones(t2.shape) * -49.963
        for i in range(t2.shape[-1] - 1):
            x2[i + 1] = rk4(f, x2[i], t2[i], t2[i + 1] - t2[i])

        sol1 = {'x': x1, 'h': np.zeros(x1.shape), 'V': np.zeros(x1.shape), 'gamma': np.zeros(x1.shape), 'alpha': np.zeros(x1.shape)}
        sol2 = {'x': x2, 'h': np.zeros(x2.shape), 'V': np.zeros(x2.shape), 'gamma': np.zeros(x2.shape), 'alpha': np.zeros(x2.shape)}

        t_refined = refineSolution(sol1, sol2, t1, t2, tol=tol, p=4)
        if t_refined.shape == t1.shape:
            print("Solution found after", iteration, " iterations.") 
            break
        else:
            t1 = t_refined
            t2 = reduceIntervalBy2(t1)

    x1 = np.ones(t1.shape) * -49.963
    for i in range(t1.shape[-1] -1):
        x1[i + 1] = rk4(f, xk=x1[i], tk=t1[i], dt=(t1[i + 1] - t1[i]))

    fig, axs = plt.subplots(3, 1)
    t_grid = np.linspace(0, t_final, 200)
    axs[0].plot(t_grid, A_original(t_grid))
    axs[0].scatter(t_refined, x1)
    axs[1].scatter(np.linspace(0, 1, N), np.zeros(N))
    axs[2].scatter(t_refined, np.zeros(t_refined.shape))
    axs[1].set_yticks([])    
    axs[2].set_yticks([])
    plt.show()

"""
Discretisation refinement for B_piecewise
"""
def test_4():
    N = 3
    t_final = 4600
    max_iter = 10
    tol = 10**-5
    t1 = np.linspace(0, t_final, N)
    t2 = np.linspace(0, t_final, 2*N - 1)

    x = ca.MX.sym('x')
    t = ca.MX.sym('t')

    xdot = ca.gradient(B_piecewise(t), t)
    f = ca.Function('f', [x, t], [xdot])
    
    for iteration in range(max_iter):
        print("iteration ", iteration + 1)
        x1 = np.zeros(t1.shape)
        for i in range(t1.shape[-1]-1):
            x1[i + 1] = rk4(f, xk=x1[i], tk=t1[i], dt=(t1[i + 1] - t1[i]))

        x2 = np.zeros(t2.shape) 
        for i in range(t2.shape[-1] - 1):
            x2[i + 1] = rk4(f, x2[i], t2[i], t2[i + 1] - t2[i])

        sol1 = {'x': x1, 'h': np.zeros(x1.shape), 'V': np.zeros(x1.shape), 'gamma': np.zeros(x1.shape), 'alpha': np.zeros(x1.shape)}
        sol2 = {'x': x2, 'h': np.zeros(x2.shape), 'V': np.zeros(x2.shape), 'gamma': np.zeros(x2.shape), 'alpha': np.zeros(x2.shape)}

        t_refined = refineSolution(sol1, sol2, t1, t2, tol=tol, p=4)
        if t_refined.shape == t1.shape:
            print("Solution found after", iteration, " iterations.") 
            break
        else:
            t1 = t_refined
            t2 = reduceIntervalBy2(t1)

    x1 = np.zeros(t1.shape)
    for i in range(t1.shape[-1] -1):
        x1[i + 1] = rk4(f, xk=x1[i], tk=t1[i], dt=(t1[i + 1] - t1[i]))

    fig, axs = plt.subplots(3, 1)
    t_grid = np.linspace(0, t_final, 200)
    axs[0].plot(t_grid, B_piecewise(t_grid))
    axs[0].scatter(t_refined, x1)
    axs[1].scatter(np.linspace(0, 1, N), np.zeros(N))
    axs[2].scatter(t_refined, np.zeros(t_refined.shape))
    axs[1].set_yticks([])    
    axs[2].set_yticks([])
    plt.show()


"""
Discretisation refinement for A_piecewise
"""
def test_3():
    t_final = 4600
    max_iter = 10
    N = 2
    tol = 10**-5
    t1 = np.linspace(0, t_final, N)
    t2 = np.linspace(0, t_final, 2*N - 1)

    x = ca.MX.sym('x')
    t = ca.MX.sym('t')

    xdot = ca.gradient(A_piecewise(t), t)
    f = ca.Function('f', [x, t], [xdot])
    
    for iteration in range(max_iter):
        print("iteration ", iteration + 1)
        x1 = np.ones(t1.shape) * -49.963
        for i in range(t1.shape[-1]-1):
            x1[i + 1] = rk4(f, xk=x1[i], tk=t1[i], dt=(t1[i + 1] - t1[i]))

        x2 = np.ones(t2.shape) * -49.963
        for i in range(t2.shape[-1] - 1):
            x2[i + 1] = rk4(f, x2[i], t2[i], t2[i + 1] - t2[i])

        sol1 = {'x': x1, 'h': np.zeros(x1.shape), 'V': np.zeros(x1.shape), 'gamma': np.zeros(x1.shape), 'alpha': np.zeros(x1.shape)}
        sol2 = {'x': x2, 'h': np.zeros(x2.shape), 'V': np.zeros(x2.shape), 'gamma': np.zeros(x2.shape), 'alpha': np.zeros(x2.shape)}

        t_refined = refineSolution(sol1, sol2, t1, t2, tol=tol, p=4)
        if t_refined.shape == t1.shape:
            print("Solution found after", iteration, " iterations.") 
            break
        else:
            t1 = t_refined
            t2 = reduceIntervalBy2(t1)

    x1 = np.ones(t1.shape) * -49.963
    for i in range(t1.shape[-1] -1):
        x1[i + 1] = rk4(f, xk=x1[i], tk=t1[i], dt=(t1[i + 1] - t1[i]))

    fig, axs = plt.subplots(3, 1)
    t_grid = np.linspace(0, t_final, 200)
    axs[0].plot(t_grid, A_piecewise(t_grid))
    axs[0].scatter(t_refined, x1)
    axs[1].scatter(np.linspace(0, 1, N), np.zeros(N))
    axs[2].scatter(t_refined, np.zeros(t_refined.shape))
    axs[1].set_yticks([])    
    axs[2].set_yticks([])
    plt.show()

"""
Discretisation refinement for B_cosh
"""
def test_2():
    N = 3
    t_final = 4600
    max_iter = 10
    N = 3
    tol = 10**-5
    t1 = np.linspace(0, t_final, N)
    t2 = np.linspace(0, t_final, 2*N - 1)

    x = ca.MX.sym('x')
    t = ca.MX.sym('t')

    xdot = dx_B_cosh(t)
    f = ca.Function('f', [x, t], [xdot])
    
    for iteration in range(max_iter):
        print("iteration ", iteration + 1)
        x1 = np.zeros(t1.shape)
        for i in range(t1.shape[-1]-1):
            x1[i + 1] = rk4(f, xk=x1[i], tk=t1[i], dt=(t1[i + 1] - t1[i]))

        x2 = np.zeros(t2.shape) 
        for i in range(t2.shape[-1] - 1):
            x2[i + 1] = rk4(f, x2[i], t2[i], t2[i + 1] - t2[i])

        sol1 = {'x': x1, 'h': np.zeros(x1.shape), 'V': np.zeros(x1.shape), 'gamma': np.zeros(x1.shape), 'alpha': np.zeros(x1.shape)}
        sol2 = {'x': x2, 'h': np.zeros(x2.shape), 'V': np.zeros(x2.shape), 'gamma': np.zeros(x2.shape), 'alpha': np.zeros(x2.shape)}

        t_refined = refineSolution(sol1, sol2, t1, t2, tol=tol, p=4)
        if t_refined.shape == t1.shape:
            print("Solution found after", iteration, " iterations.") 
            break
        else:
            t1 = t_refined
            t2 = reduceIntervalBy2(t1)

    x1 = np.zeros(t1.shape)
    for i in range(t1.shape[-1] -1):
        x1[i + 1] = rk4(f, xk=x1[i], tk=t1[i], dt=(t1[i + 1] - t1[i]))

    fig, axs = plt.subplots(3, 1)
    t_grid = np.linspace(0, t_final, 200)
    axs[0].plot(t_grid, B_cosh(t_grid))
    axs[0].scatter(t_refined, x1)
    axs[1].scatter(np.linspace(0, 1, N), np.zeros(N))
    axs[2].scatter(t_refined, np.zeros(t_refined.shape))
    axs[1].set_yticks([])    
    axs[2].set_yticks([])
    plt.show()

"""
Discretisation refinement for A_cosh
"""
def test_1():
    t_final = 4600
    max_iter = 10
    N = 2
    tol = 10**-5
    t1 = np.linspace(0, t_final, N)
    t2 = np.linspace(0, t_final, 2*N - 1)

    x = ca.MX.sym('x')
    t = ca.MX.sym('t')

    xdot = dx_A_cosh(t)
    f = ca.Function('f', [x, t], [xdot])
    
    for iteration in range(max_iter):
        print("iteration ", iteration + 1)
        x1 = np.ones(t1.shape) * -49.963
        for i in range(t1.shape[-1]-1):
            x1[i + 1] = rk4(f, xk=x1[i], tk=t1[i], dt=(t1[i + 1] - t1[i]))

        x2 = np.ones(t2.shape) * -49.963
        for i in range(t2.shape[-1] - 1):
            x2[i + 1] = rk4(f, x2[i], t2[i], t2[i + 1] - t2[i])

        sol1 = {'x': x1, 'h': np.zeros(x1.shape), 'V': np.zeros(x1.shape), 'gamma': np.zeros(x1.shape), 'alpha': np.zeros(x1.shape)}
        sol2 = {'x': x2, 'h': np.zeros(x2.shape), 'V': np.zeros(x2.shape), 'gamma': np.zeros(x2.shape), 'alpha': np.zeros(x2.shape)}

        t_refined = refineSolution(sol1, sol2, t1, t2, tol=tol, p=4)
        if t_refined.shape == t1.shape:
            print("Solution found after", iteration, " iterations.") 
            break
        else:
            t1 = t_refined
            t2 = reduceIntervalBy2(t1)

    x1 = np.ones(t1.shape) * -49.963
    for i in range(t1.shape[-1] -1):
        x1[i + 1] = rk4(f, xk=x1[i], tk=t1[i], dt=(t1[i + 1] - t1[i]))

    fig, axs = plt.subplots(3, 1)
    t_grid = np.linspace(0, t_final, 200)
    axs[0].plot(t_grid, A_cosh(t_grid))
    axs[0].scatter(t_refined, x1)
    axs[1].scatter(np.linspace(0, 1, N), np.zeros(N))
    axs[2].scatter(t_refined, np.zeros(t_refined.shape))
    axs[1].set_yticks([])    
    axs[2].set_yticks([])
    plt.show()




def rk4(f, xk, tk, dt):
    k1 = f(xk, tk)
    k2 = f(
        xk + dt / 2 * k1, tk + dt / 2
    )
    k3 = f(
        xk + dt / 2 * k2, tk + dt / 2
    )
    k4 = f(xk + dt * k3, tk + dt)
    return xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def A_original(x_):
    a = 6e-8  
    b = -4e-11  

    A1 = -50 + a * x_**3 + b * x_**4
    A2 = 0.025 * (x_ - 2300)
    A3 = 50 - a * (4600 - x_) ** 3 - b * (4600 - x_) ** 4
    A4 = 50
    return ca.if_else(
        x_ <= 500, A1, ca.if_else(x_ <= 4100, A2, ca.if_else(x_ <= 4600, A3, A4))
    )

def B_original(x_):
    c = -np.log(25 / 30.6) * 1e-12  
    d = -8.02881e-8  
    e = 6.28083e-11

    B1 = d * x_**3 + e * x_**4
    B2 = -51 * ca.exp(ca.fmin(-c * (x_ - 2300) ** 4, 30))
    B3 = d * (4600 - x_) ** 3 + e * (4600 - x_) ** 4
    B4 = 0
    return ca.if_else(
        x_ <= 500, B1, ca.if_else(x_ <= 4100, B2, ca.if_else(x_ <= 4600, B3, B4))
    )

if __name__ == "__main__":
    main()


