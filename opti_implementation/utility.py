"""
Some utility functions.
"""
from .multiple_planes_solution_advanced import solve_multiple_planes_ocp_advanced

import numpy as np
import matplotlib.pyplot as plt
from time import time

"""
Calculates the mean run-time of two model set-ups.
---
## Params:
 - iterations : number of iterations to approximate mean time.
 - k_values : k_values used for the calculation.
 - integrators : array of length 2 containing the used integrator for each model.
 - windmodels : array of length 2 containing the used windmodel for each model.
---
## Return:
 - Returns the mean run-time for both model set-ups.
"""
def timeModels(iterations, k_values, integrators, windmodels):
    sol_0 = solve_multiple_planes_ocp_advanced(k_values = k_values, windmodel=windmodels[0], integrator=integrators[0], verbose=False)

    times = np.zeros((2, iterations))
    for iter in range(iterations):
        selection_index = np.random.randint(0, 2)
        start_time = time()
        sol = solve_multiple_planes_ocp_advanced(k_values = k_values, windmodel=windmodels[selection_index], integrator=integrators[selection_index], verbose=False)
        times[selection_index, iter] = time() - start_time

        selection_index = (selection_index + 1) % 2
        start_time = time()
        sol = solve_multiple_planes_ocp_advanced(k_values = k_values, windmodel=windmodels[selection_index], integrator=integrators[selection_index], verbose=False)
        times[selection_index, iter] = time() - start_time

    average_time_1 = times[0].sum() / iterations
    average_time_2 = times[1].sum() / iterations
    print("average time 1:", average_time_1)
    print("average time 2:", average_time_2)

"""
Plots each state variable, as well as the control variable on a designated graph,
making easy comparing of different solutions possible.
---
## Params:
 - solutions : array of solutions that have to be plotted.
"""
def plotSolutions(solutions):
    figure, ax = plt.subplots(3, 2)
    legends = np.empty((3, 2, len(solutions)), dtype=np.dtypes.StrDType)
    for k, solution in enumerate(solutions): 
        t = np.linspace(0, 40, len(solution["x"]))  
        t_u = np.linspace(0, 40, len(solution["u"])) 
        
        x_k = solution["x"]
        ax[0, 0].plot(t, x_k)
        legends[0, 0, k] = "x_" + str(k)

        h_k = solution["h"]
        ax[1, 0].plot(t, h_k)
        legends[1, 0, k] = 'h_' + str(k)

        v_k = solution["V"]
        ax[2, 0].plot(t, v_k)
        legends[2, 0, k] = 'V_' + str(k)

        gamma_k = solution["gamma"]
        ax[0, 1].plot(t, gamma_k)
        legends[0 ,1, k] = 'gamma_' + str(k)

        alpha_k = solution["alpha"]
        ax[1, 1].plot(t, alpha_k)
        legends[1, 1, k] = 'alpha_' + str(k)

        u_k = solution["u"]
        ax[2, 1].step(t_u, u_k)
        legends[2, 1, k] = 'u_' + str(k)
    ax[0, 0].legend(legends[0, 0, :])
    ax[0, 1].legend(legends[0, 1, :])
    ax[1, 0].legend(legends[1, 0, :])
    ax[1, 1].legend(legends[1, 1, :])
    ax[2, 0].legend(legends[2, 0, :])
    ax[2, 1].legend(legends[2, 1, :])
    plt.show()

"""
Splits solution of the multiple plain set-up into designated single plane solutions.
---
## Params:
 - solution : original multy plane solution.
---
## Return:
 - Returns an array of single plane solutions.
"""
def splitSolution(solution):
    if solution['x'].ndim == 1: #Only 1 solution
        return solution 
    else:
        n_solutions = solution['x'].shape[0]
        solutions = []
        for i in range(n_solutions):
            solutions_entry = {
                'x': solution['x'][i, :],
                'h': solution['h'][i, :],
                'V': solution['V'][i, :],
                'gamma': solution['gamma'][i, :],
                'alpha': solution['alpha'][i, :],
                'u': solution['u'],
                'w': solution['w'],
            }
            solutions.append(solutions_entry)
        return solutions