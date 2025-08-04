[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![CasADi](https://img.shields.io/badge/CasADi-3.7.0-blue.svg)](https://web.casadi.org/)
[![Chaospy](https://img.shields.io/badge/Chaospy-4.3.20-orange.svg)](https://chaospy.readthedocs.io/)
[![NumPy](https://img.shields.io/badge/Numpy-2.3.1-orange.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.3-yellow.svg)](https://matplotlib.org/)

# Case Studies S3: Abort Landing in case of Uncertain Windshear

This repository contains all the code and data developed during the Case Studies S3 project at Technical University of Munich (TUM).

The goal of this study is to investigate **optimal abort landing strategies under uncertain windshear conditions**, using **stochastic optimal control**, **wind modeling**, and **polynomial chaos expansion (PCE)**.

---

## Project Overview

Abort landings are critical maneuvers that must be executed safely, even in the presence of unpredictable windshear. In this project, we model wind uncertainty and embed it into an optimal control framework to develop robust abort landing strategies.

### Key Features

- Wind modeling with tunable **intensity** parameter \( k \) and **width** parameter \( s \)
- Uncertainty quantification using **Polynomial Chaos Expansion (PCE)**, built with `Chaospy`
- Optimal control problem formulated and solved using **CasADi**
- Adaptive time grids and robust trajectory measurement strategies

---

## Folder Structure

```
case_studies_s3/
  ├── A320neo_model/ # Aircraft dynamics and physical constraints
  ├── Determenistic_OCP/ # Determenistic optimization
  │   ├── solvers_and_functions_package # Python package containing solvers and other related functions
  │   ├── Determenistic_OCP.ipynb # Exemple code on how to use the solvers
  ├── PCE/ # PCE-based stochastic optimization
  │   ├── All_in_Poster.ipynb # Summary of progress for poster
  │   ├── numerical_experiments.ipynb # Main experimental notebook
  │   ├── mc_failure_probability.py # MC-based failure probability estimation
  │   ├── sus_new.py # Subset simulation (SuS) implementation
  │   ├── visualization.py # Plotting utilities
  ├── example_problems/ # Minimal test problems
  ├── step_size_control/ # Adaptive integration / optimization modules
```

---

## Contributors

- **Chenhong Lin** — Wind modeling, PCE design, and stochastic control strategy
- **Miaowen Dong** — Code implementation of the stochastic optimal control problem under uncertain windshear and numerical experimentation
- **Irma Svensson** — Code implementation of the determenistic optimal control problem

---

## License

This repository is for academic and non-commercial use only. Please contact the authors for reuse or collaboration.
