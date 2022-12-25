import json
import time

from typing import Callable, Dict

import numpy as np

from parzen_estimator import NumericalParzenEstimator, NumericalUniform

import optuna
from optuna.importance import FanovaImportanceEvaluator


optuna.logging.set_verbosity(optuna.logging.CRITICAL)
D = 4
GAMMA = 0.1
N_SEEDS = 10
N_GRIDS = 1000
LB, UB = -5, 5


def func(X: np.ndarray) -> np.ndarray:
    assert X.shape[0] == D
    W = np.asarray([5 ** -i for i in range(D)])
    Z = np.zeros(X.shape[-1])
    for d in range(D):
        mask = np.abs(X[d]) < 1
        Z += W[(mask + d) % D] * X[d] ** 2

    return Z


def analyze_by_ours(X: np.ndarray, F: np.ndarray, *args, **kwargs) -> Dict[str, float]:
    order = np.argsort(F)
    n_samples = F.size
    dx = np.linspace(LB, UB, N_GRIDS + 1)
    X_rounded = np.round(((X + LB) / (UB - LB)) * N_GRIDS) / N_GRIDS * (UB - LB) - LB
    u = NumericalUniform(lb=LB, ub=UB)
    hpi_dict = {}
    for d in range(D):
        x = X_rounded[d]
        pe = NumericalParzenEstimator(x[order[:int(GAMMA * n_samples)]], lb=LB, ub=UB, compress=True)
        U = u(dx)
        hpi = U @ ((pe(dx) / u(dx) - 1) ** 2)
        hpi *= GAMMA ** 2 / np.sum(U)
        hpi_dict[f"x{d+1}"] = hpi

    return hpi_dict


def analyze_by_optuna(X: np.ndarray, F: np.ndarray, seed: int) -> Dict[str, float]:
    study = optuna.create_study()
    study.add_trials([
        optuna.trial.create_trial(
            params={f"x{d+1}": x for d, x in enumerate(xs)},
            distributions={f"x{d+1}": optuna.distributions.FloatDistribution(LB, UB) for d in range(D)},
            value=f,
        )
        for xs, f in zip(X.T, F)
    ])
    result = optuna.importance.get_param_importances(study, evaluator=FanovaImportanceEvaluator(seed=seed))
    return dict(result)


def update(
    x: np.ndarray,
    f: np.ndarray,
    size: int,
    seed: int,
    results: Dict,
    analyze_fn: Callable,
) -> None:
    start = time.time()
    result = analyze_fn(x, f, seed)
    runtime = time.time() - start
    result_str = {k: float(f"{v:.2e}") for k, v in result.items()}
    print(f"Finish {size=}, {seed=} in {runtime:.2e} seconds with the result: {result_str}")

    for k, v in result.items():
        results[size][k].append(v)

    results[size]["runtime"].append(runtime)


def validate(sizes: np.ndarray, analyze_fn: Callable) -> Dict:
    results = {size: {k: [] for k in [f"x{d+1}" for d in range(D)] + ["runtime"]} for size in sizes}
    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed)
        X = rng.random((D, sizes[-1])) * (UB - LB) + LB
        F = func(X)
        for size in sizes:
            assert size <= F.size
            x, f = X[:, :size], F[:size]
            update(x, f, size, seed, results, analyze_fn=analyze_fn)

    return results


def validate_ours() -> Dict:
    P = 8
    sizes = np.hstack([[10 ** i, 3 * 10 ** i] for i in range(1, P)] + [10 ** P])
    return validate(sizes, analyze_by_ours)


def validate_optuna() -> Dict:
    P = 4
    sizes = np.hstack([[10 ** i, 3 * 10 ** i] for i in range(1, P)] + [10 ** P])
    return validate(sizes, analyze_by_optuna)


if __name__ == "__main__":
    results = validate_ours()
    json.dump({str(k): v for k, v in results.items()}, open("validate-ours.json", "w"), indent=4)
