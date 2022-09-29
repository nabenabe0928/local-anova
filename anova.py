from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from parzen_estimator import NumericalParzenEstimator, NumericalUniform

from scipy.stats import rankdata


ParzenEstimatorType = Union[NumericalParzenEstimator, NumericalUniform]


def perason_divergence(
    pe_local: ParzenEstimatorType,
    pe_global: ParzenEstimatorType,
    n_samples: int,
    rng: np.random.RandomState,
) -> float:
    """
    Compute the Pearson divergence by Monte-Carlo method.

    Args:
        pe_local (ParzenEstimatorType):
            The Parzen estimator that implicitly defines a local space.
        pe_global (ParzenEstimatorType):
            The Parzen estimator that implicitly defines a global space.
        n_samples (int):
            The number of samples used for Monte-Carlo method.
        rng (np.random.RandomState):
            The random number generator.

    Returns:
        divergence (float):
            Pearson divergence between pe_local and pe_global.
            We omit gamma as this number is constant for all dimensions.
    """
    x = pe_global.sample(rng=rng, n_samples=n_samples)
    # we omit (gamma^\prime / gamma) ** 2
    return np.mean((pe_local(x) / pe_global(x) - 1) ** 2)


def global_hpi(
    observations: Dict[str, np.ndarray],
    bounds: Dict[str, Tuple[float, float]],
    param_names: List[str],
    obj_name: str,
    gamma: float,
    n_samples: int = 100,
    minimize: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute the global hyperparameter importance.

    Args:
        observations (Dict[str, np.ndarray]):
            Observations obtained by an HPO.
            Objective function values must also be included.
            observations[param name] = array of the observations in the param.
        bounds (Dict[str, Tuple[float, float]]):
            The lower/upper bounds of each param.
            bounds[param name] = (lower bound of the param, upper bound of the param).
        param_names (List[str]):
            A list of hyperparameter names.
        obj_name (str):
            The name of the objective function.
        gamma (float):
            The quantile value that defines the target domain.
        n_samples (int):
            The number of samples used in Monte-Carlo method.
        minimize (bool):
            Whether the objective function is better when it is smaller.
        seed (Optional[int]):
            The seed for a random number generator.

    Returns:
        hpi_dict (Dict[str, float]):
            The hyperparameter importance for each param.
            hpi_dict[param name] = the importance of this param in the global space.
    """
    assert 0 < gamma <= 1
    sign = 2 * minimize - 1
    loss_vals = observations[obj_name] * sign
    rank = rankdata(loss_vals)

    rng = np.random.RandomState(seed)
    N = loss_vals.size
    gamma_set_mask = rank < N * gamma
    hpi_dict: Dict[str, float] = {}
    for param_name in param_names:
        lb, ub = bounds[param_name]
        params = observations[param_name]
        u = NumericalUniform(lb=lb, ub=ub)
        pe = NumericalParzenEstimator(samples=params[gamma_set_mask], lb=lb, ub=ub)
        hpi_dict[param_name] = perason_divergence(
            pe_local=pe,
            pe_global=u,
            n_samples=n_samples,
            rng=rng,
        )

    return hpi_dict


def local_hpi(
    observations: Dict[str, np.ndarray],
    bounds: Dict[str, Tuple[float, float]],
    param_names: List[str],
    obj_name: str,
    gamma_local_top: float,
    gamma_local: float,
    n_samples: int = 100,
    minimize: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute the global hyperparameter importance.

    Args:
        observations (Dict[str, np.ndarray]):
            Observations obtained by an HPO.
            Objective function values must also be included.
            observations[param name] = array of the observations in the param.
        bounds (Dict[str, Tuple[float, float]]):
            The lower/upper bounds of each param.
            bounds[param name] = (lower bound of the param, upper bound of the param).
        param_names (List[str]):
            A list of hyperparameter names.
        obj_name (str):
            The name of the objective function.
        gamma_local_top (float):
            The quantile value that defines the local space.
            For example, when gamma_local_top = 0.01 and gamma_local = 0.1,
            we measure the hpi to obtain the top-0.01 quantile in the "global" space
            from the top-0.1 local space.
            We can also consider it as:
                we measure the hpi to obtain the top-0.1 quantile in the "local" space.
        gamma_local (float):
            The quantile value that defines the target domain in the local space.
        n_samples (int):
            The number of samples used in Monte-Carlo method.
        minimize (bool):
            Whether the objective function is better when it is smaller.
        seed (Optional[int]):
            The seed for a random number generator.

    Returns:
        hpi_dict (Dict[str, float]):
            The hyperparameter importance for each param.
            hpi_dict[param name] = the importance of this param in the local space.
    """
    assert 0 < gamma_local_top < gamma_local <= 1
    sign = 2 * minimize - 1
    loss_vals = observations[obj_name] * sign
    rank = rankdata(loss_vals)

    rng = np.random.RandomState(seed)
    N = loss_vals.size
    local_space_mask = rank < N * gamma_local
    top_in_local_space_mask = rank < N * gamma_local_top
    hpi_dict: Dict[str, float] = {}
    for param_name in param_names:
        lb, ub = bounds[param_name]
        params = observations[param_name]
        pe_local = NumericalParzenEstimator(samples=params[local_space_mask], lb=lb, ub=ub)
        pe_local_top = NumericalParzenEstimator(samples=params[top_in_local_space_mask], lb=lb, ub=ub)
        hpi_dict[param_name] = perason_divergence(
            pe_local=pe_local_top,
            pe_global=pe_local,
            n_samples=n_samples,
            rng=rng,
        )

    return hpi_dict
