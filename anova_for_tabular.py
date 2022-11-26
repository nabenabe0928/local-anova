from typing import Any, Dict, List, Optional, Union

import numpy as np

import ujson as json

import matplotlib.pyplot as plt

import pandas as pd

from parzen_estimator import (
    CategoricalParzenEstimator,
    CategoricalUniform,
    NumericalParzenEstimator,
    NumericalUniform
)

from scipy.stats import rankdata

from index_to_config import convert_multiple


ParzenEstimatorType = Union[
    CategoricalParzenEstimator,
    CategoricalUniform,
    NumericalParzenEstimator,
    NumericalUniform,
]
SEARCH_SPACE = dict(
    LearningRate=(0.001, 0.01, 0.1, 1.0),
    WeightDecay=(0.00001, 0.0001, 0.001, 0.01),
    N=(1, 3, 5),
    W=(4, 8, 16),
    Activation=("ReLU", "Hardswish", "Mish"),
    TrivialAugment=(True, False),
    Op1=(0, 1, 2, 3, 4),
    Op2=(0, 1, 2, 3, 4),
    Op3=(0, 1, 2, 3, 4),
    Op4=(0, 1, 2, 3, 4),
    Op5=(0, 1, 2, 3, 4),
    Op6=(0, 1, 2, 3, 4)
)


def perason_divergence(
    pe_local: ParzenEstimatorType,
    pe_global: ParzenEstimatorType,
    choices: np.ndarray,
) -> float:
    """
    Compute the Pearson divergence by Monte-Carlo method.

    Args:
        pe_local (ParzenEstimatorType):
            The Parzen estimator that implicitly defines a local space.
        pe_global (ParzenEstimatorType):
            The Parzen estimator that implicitly defines a global space.
        choices (np.ndarray):
            The choices of the config.

    Returns:
        divergence (float):
            Pearson divergence between pe_local and pe_global.
            We omit gamma as this number is constant for all dimensions.
    """
    pl = pe_local(choices)
    pg = pe_global(choices)
    # we omit (gamma^\prime / gamma) ** 2
    return np.average((pl / pg - 1) ** 2, weights=pg)


def plot_pe(pe: ParzenEstimatorType, name: str, choices: np.ndarray, quantile: float) -> None:
    indices = np.arange(len(choices))

    _, ax = plt.subplots()
    ax.set_title(name)
    buffer = 0.5
    ax.hlines(1.0 / len(choices), -buffer, len(choices) - 1 + buffer, linestyle="--", color="red", label="Uniform")
    ax.bar(indices, pe(indices), tick_label=choices, label=f"Top {quantile*100:.2f}%")
    ax.grid(axis="y")
    ax.set_xlim(-buffer, len(choices) - 1 + buffer)
    ax.legend()
    plt.show()


def fetch_top_configs(quantile: float, ranks: np.ndarray) -> pd.DataFrame:
    top_indices = np.arange(size)[ranks <= size * quantile]
    top_configs = convert_multiple(top_indices, return_by_index=True)
    return top_configs


def get_parzen_estimator(
    name: str,
    choices: List[Any],
    configs: Optional[pd.DataFrame],
    is_cat: bool,
) -> ParzenEstimatorType:
    n_choices = len(choices)
    if is_cat:
        if configs is None:
            return CategoricalUniform(n_choices=n_choices)
        else:
            samples = configs[name].to_numpy()
            return CategoricalParzenEstimator(samples=samples, n_choices=n_choices, top=1.0)
    else:
        if configs is None:
            return NumericalUniform(lb=0, ub=n_choices - 1, dtype=np.int32, q=1)
        else:
            samples = configs[name].to_numpy()
            return NumericalParzenEstimator(
                samples=samples, lb=0, ub=n_choices - 1, compress=True, dtype=np.int32, q=1,
            )


def compute_hpi(
    ranks: np.ndarray,
    local_quantile: pd.DataFrame,
    global_quantile: Optional[float] = None,
) -> Dict[str, float]:

    global_configs = None if global_quantile is None else fetch_top_configs(global_quantile, ranks)
    local_configs = fetch_top_configs(local_quantile, ranks)

    numeric_params = ["LearningRate", "WeightDecay", "N", "W"]
    hpi = {}
    if global_configs is None:
        print(f"Calculate the global HPI to achieve the top {100 * local_quantile:.2f}%")
    else:
        print(
            f"Calculate the local HPI to achieve the top {100 * local_quantile:.2f}% from "
            f"the top {100 * global_quantile:.2f}%"
        )
    for name, choices in SEARCH_SPACE.items():
        is_cat = name not in numeric_params
        pe_local = get_parzen_estimator(name=name, choices=choices, configs=local_configs, is_cat=is_cat)
        pe_global = get_parzen_estimator(name=name, choices=choices, configs=global_configs, is_cat=is_cat)

        # plot_pe(pe_local, name=name, choices=choices, quantile=quantile)
        hpi[name] = perason_divergence(pe_local, pe_global, choices=np.arange(len(choices)))

    return hpi


if __name__ == "__main__":
    print("Load data")
    loss_vals = np.asarray(json.load(open("cifar10.json")))

    size = loss_vals.size
    print("Calculate ranking")
    ranks = rankdata(loss_vals)
    hpi = compute_hpi(ranks, local_quantile=0.002, global_quantile=0.01)
    total_hpi = sum(hpi.values())
    for name, hpi in hpi.items():
        print(f"{name}: {hpi / total_hpi * 100:.2f}%")
