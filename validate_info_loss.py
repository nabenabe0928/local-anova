import time

from typing import Dict

import numpy as np

from parzen_estimator import NumericalParzenEstimator, NumericalUniform

import optuna
from optuna.importance import FanovaImportanceEvaluator


optuna.logging.set_verbosity(optuna.logging.CRITICAL)
D = 4
N = 10 ** 5
gamma = 0.1
LB, UB = -5, 5


def func(X: np.ndarray) -> np.ndarray:
    assert X.shape[0] == D
    W = np.asarray([5 ** -i for i in range(D)])
    Z = np.zeros(X.shape[-1])
    for d in range(D):
        mask = np.abs(X[d]) < 1
        Z += W[(mask + d) % D] * X[d] ** 2

    return Z


def analyze_by_ours(X: np.ndarray, order: np.ndarray) -> Dict[str, float]:
    X_rounded = np.round(((X + LB) / (UB - LB)) * n_grids) / n_grids * (UB - LB) - LB
    u = NumericalUniform(lb=LB, ub=UB)
    hpi_dict = {}
    for d in range(D):
        x = X_rounded[d]
        pe = NumericalParzenEstimator(x[order[:int(gamma * N)]], lb=LB, ub=UB, compress=True)
        U = u(dx)
        hpi = U @ ((pe(dx) / u(dx) - 1) ** 2)
        hpi *= gamma ** 2 / np.sum(U)
        hpi_dict[f"x{d+1}"] = hpi

    return hpi_dict


def analyze_by_optuna(X: np.ndarray, F: np.ndarray) -> None:
    study = optuna.create_study()
    study.add_trials([
        optuna.trial.create_trial(
            params={f"x{d+1}": x for d, x in enumerate(xs)},
            distributions={f"x{d+1}": optuna.distributions.FloatDistribution(LB, UB) for d in range(D)},
            value=f,
        )
        for xs, f in zip(X.T, F)
    ])
    result = optuna.importance.get_param_importances(study, evaluator=FanovaImportanceEvaluator())
    return dict(result)


if __name__ == "__main__":
    n_grids = 10
    dx = np.linspace(LB, UB, n_grids + 1)
    # order = np.argsort(F)
    # print(analyze_by_ours(X, order))

    for seed in range(10):
        rng = np.random.RandomState(seed)
        X = rng.random((D, N)) * (UB - LB) + LB
        F = func(X)
        for size in [10, 3 * 10, 10 ** 2, 3 * 10 ** 2, 10 ** 3, 3 * 10 ** 3, 10 ** 4, 3 * 10 ** 4, 10 ** 5]:
            start = time.time()
            result = analyze_by_optuna(X[:size], F[:size])
            print(f"Finish {size=}, {seed=} in {time.time() - start:.3e} seconds with the result: {result}")
