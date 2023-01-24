from typing import Dict, Tuple

import numpy as np

import pandas as pd

from parzen_estimator import (
    CategoricalParzenEstimator,
    CategoricalUniform,
    NumericalParzenEstimator,
    NumericalUniform,
)


def get_pdf_vals(
    X_local: pd.DataFrame,
    X_global: pd.DataFrame,
    hp_name: str,
    search_space: Dict[str, Tuple],
    categoricals: Dict[str, bool] = {},
) -> Tuple[np.ndarray, np.ndarray]:
    x_local = X_local[hp_name].to_numpy()
    x_global = X_global[hp_name].to_numpy() if X_global is not None else None
    n_choices = len(search_space[hp_name])
    kwargs = {}
    dx = np.arange(n_choices)
    if categoricals.get(hp_name, False):
        kwargs.update(top=1.0, n_choices=n_choices)
        pe_local = CategoricalParzenEstimator(samples=x_local, **kwargs)
        pe_global = (
            CategoricalUniform(n_choices=n_choices)
            if x_global is None
            else CategoricalParzenEstimator(samples=x_global, **kwargs)
        )
    else:
        kwargs.update(lb=0, ub=n_choices - 1, q=1, dtype=int)
        pe_local = NumericalParzenEstimator(samples=x_local, compress=True, **kwargs)
        pe_global = (
            NumericalUniform(**kwargs)
            if x_global is None
            else NumericalParzenEstimator(samples=x_global, compress=True, **kwargs)
        )

    return pe_local(dx), pe_global(dx)


def analyze(
    X: pd.DataFrame,
    F: np.ndarray,
    search_space: Dict[str, Tuple],
    gamma_local: float,
    gamma_global: float = 1.0,
    categoricals: Dict[str, bool] = {},
) -> Dict[str, float]:
    """
    Args:
        X (pd.DataFrame):
            The hyperparameter dataframe.
            The hyperparameter values must be grid indices.
            For example, if param1 takes (0.1, 0.3) among the space of (0.1, 0.2, 0.3),
            then the corresponding grid indices will be (0, 2).
        F (np.ndarray):
            The performance metric (lower is better).
            The order must match the indices of X.
        search_space (Dict[str, Tuple]):
            The search space of the target of analysis.
        gamma_local (float):
            The quantile for a local space.
        gamma_global (float):
            The quantile for a global space.
        categoricals (Dict[str, bool]):
            Whether the corresponding hyperparameter is categorical or not.

    Returns:
        hpi_dict (Dict[str, float]):
            The dict of HPI.
    """
    # NOTE: F must be "lower is better"
    order = np.argsort(F)
    hpi_dict = {}
    hpi_sum = 0
    X_local = X.iloc[order[: int(gamma_local * F.size)]]
    X_global = (
        X.iloc[order[: int(gamma_global * F.size)]] if gamma_global < 1.0 else None
    )

    for hp_name in X.columns:
        pdf_local, pdf_global = get_pdf_vals(
            X_local=X_local,
            X_global=X_global,
            hp_name=hp_name,
            search_space=search_space,
            categoricals=categoricals
        )
        hpi_dict[hp_name] = pdf_global @ ((pdf_local / pdf_global - 1) ** 2)
        hpi_sum += hpi_dict[hp_name]
    else:
        hpi_dict = {k: v / hpi_sum for k, v in hpi_dict.items()}

    return hpi_dict
