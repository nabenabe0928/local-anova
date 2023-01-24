from typing import Dict

from anova import global_hpi, local_hpi

import numpy as np


LB, UB = -5, 5
N = 10000
DIM = 4
rng = np.random.RandomState(42)
param_names = [f"x{d}" for d in range(DIM)]
obj_name = "loss"
bounds = {param_name: (LB, UB) for param_name in param_names}


def func(X: Dict[str, np.ndarray]) -> np.ndarray:
    weights = np.array([5**-d for d in range(DIM)])
    loss_vals = np.zeros_like(X[param_names[0]])
    for d, param_name in enumerate(param_names):
        x = X[param_name]
        weight = np.zeros_like(x)
        mask = np.abs(x) < 1
        weight[mask] = weights[(d + 1) % DIM]
        weight[~mask] = weights[d]
        loss_vals += weight * x**2

    return loss_vals


if __name__ == "__main__":
    observations = {
        param_name: rng.random(N) * (UB - LB) + LB for param_name in param_names
    }
    observations[obj_name] = func(X=observations)
    print(global_hpi(observations, bounds, param_names, obj_name, gamma=0.1))
    print(
        local_hpi(
            observations,
            bounds,
            param_names,
            obj_name,
            gamma_local_top=0.01,
            gamma_local=0.1,
        )
    )
