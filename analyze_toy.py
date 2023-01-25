from typing import Dict

import numpy as np

import pandas as pd

from utils.anova import analyze


LB, UB = -5, 5
N_SAMPLES = 10000
DIM = 4
PARAM_NAMES = [f"x{d}" for d in range(DIM)]
OBJ_NAME = "loss"
bounds = {param_name: (LB, UB) for param_name in PARAM_NAMES}


def func(X: Dict[str, np.ndarray]) -> np.ndarray:
    weights = np.array([5**-d for d in range(DIM)])
    loss_vals = np.zeros_like(X[PARAM_NAMES[0]])
    for d, param_name in enumerate(PARAM_NAMES):
        x = X[param_name]
        weight = np.zeros_like(x)
        mask = np.abs(x) < 1
        weight[mask] = weights[(d + 1) % DIM]
        weight[~mask] = weights[d]
        loss_vals += weight * x**2

    return loss_vals


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    n_disc = 1001
    choices = np.arange(n_disc)
    # For the analysis, we use the grid indices.
    X = {
        param_name: rng.choice(choices, size=N_SAMPLES, replace=True)
        for param_name in PARAM_NAMES
    }
    search_space = {f"x{d}": tuple([float(c) for c in choices]) for d in range(DIM)}

    # X must be converted into [LB, UB] for the objective function.
    F = func({k: v / (n_disc - 1) * (UB - LB) + LB for k, v in X.items()})

    print("Global HPI for the top-0.1 quantile")
    print(analyze(X=pd.DataFrame(X), F=F, search_space=search_space, gamma_local=0.1))
    print("Local HPI for the top-0.01 quantile in the top-0.1 quantile")
    print(analyze(X=pd.DataFrame(X), F=F, search_space=search_space, gamma_global=0.1, gamma_local=0.01))
