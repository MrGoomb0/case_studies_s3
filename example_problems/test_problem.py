"""
Test file that shows how the modules can be used.
"""

from casadi import *
import numpy as np
from old_NLP_implementation.wind_model import easyWindModel, windModel, windModelChebychev
from old_NLP_implementation.problem_def import problem2D
from old_NLP_implementation.integrators import cranknicholson, expEuler, rungakutta6
from old_NLP_implementation.solvers import nlpsolver, initialEstimatorUsingExpEuler
import matplotlib.pyplot as plt

"""
Problem solving
"""

T = 40
N = 3
M = 50
X0 = 0
GAMMA0 = 0
H0 = 600
ALPHA0 = 0
V0 = 100
GAMMA_F = inf
ALPHA_MAX = inf

f = problem2D(windmodel=windModelChebychev, k=1, q=2, degree=10, N_wind=1000)
F = rungakutta6(f, T, N, M)

u = np.ones(N) * 3 / 180 * pi

init_estimate = initialEstimatorUsingExpEuler(f, N, M, u, x_0=X0, h_0=H0, v_0=V0, gamma_0=GAMMA0, alpha_0=ALPHA0, t_f=T)

print("Started solving")
solution = nlpsolver(N, F, init_estimate, u, T, GAMMA_F, ALPHA_MAX)

"""
Plots
"""

u_opt = [solution[6*i + 5] for i in range(N)]
x_opt = [solution[6*i + 0] for i in range(N)]
h_opt = [solution[6*i + 1] for i in range(N)]
v_opt = [solution[6*i + 2] for i in range(N)]
gamma_opt = [solution[6*i + 3] for i in range(N)]
alpha_opt = [solution[6*i + 4] for i in range(N)]

x_opt_full = np.zeros(N * M)
h_opt_full = np.zeros(N * M)
v_opt_full = np.zeros(N * M)
gamma_opt_full = np.zeros(N * M)
alpha_opt_full = np.zeros(N * M)

F = expEuler(f , T, N*M, 1)
Tk = 0
J = 0

for i in range(N):
    x_start = x_opt[i]
    x_opt_full[i*M] = x_opt[i]
    h_start = h_opt[i]
    h_opt_full[i*M] = h_opt[i]
    v_start = v_opt[i]
    v_opt_full[i*M] = v_opt[i]
    gamma_start = gamma_opt[i]
    gamma_opt_full[i*M] = gamma_opt[i]
    alpha_start = alpha_opt[i]
    alpha_opt_full[i*M] = alpha_opt[i]

    for j in range(M-1):
        Xk = [x_opt_full[i*M + j],
              h_opt_full[i*M + j],
              v_opt_full[i*M + j],
              gamma_opt_full[i*M + j],
              alpha_opt_full[i*M + j]]
        Fk =F(x0=Xk, j0=J, t0=Tk, p=u_opt[i])
        x_opt_full[i*M + j + 1] = Fk['xf'][0]
        h_opt_full[i*M + j + 1] = Fk['xf'][1]
        v_opt_full[i*M + j + 1] = Fk['xf'][2]
        gamma_opt_full[i*M + j + 1] = Fk['xf'][3]
        alpha_opt_full[i*M + j + 1] = Fk['xf'][4]
        Tk = Fk['tf']
t_u = np.linspace(0, T, N + 1)
t = np.linspace(0, T, N * M)

plt.step(t_u, vertcat(DM.nan(1), u_opt), '-.')
plt.grid()
plt.show()
plt.plot(t, x_opt_full)
plt.grid()
plt.show()
plt.plot(t, h_opt_full)
plt.grid()
plt.show()
plt.plot(t, v_opt_full)
plt.grid()
plt.show()
plt.plot(t, gamma_opt_full)
plt.grid()
plt.show()
plt.plot(t, alpha_opt_full)
plt.grid()
plt.show()

# print(solution.shape)
