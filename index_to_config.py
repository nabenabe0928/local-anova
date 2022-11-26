from typing import Any, Dict
import itertools

import numpy as np

import pandas as pd


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
PARAM_NAMES = list(SEARCH_SPACE.keys())
BASES = [len(c) for c in reversed(SEARCH_SPACE.values())]


def convert(index: int, return_by_index: bool = False) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for b, (k, choices) in zip(BASES, reversed(SEARCH_SPACE.items())):
        target = np.arange(len(choices)) if return_by_index else choices
        result[k] = target[index % b]
        index //= b

    return result


def convert_multiple(indices: np.ndarray, return_by_index: bool = False) -> pd.DataFrame:
    result: Dict[str, Any] = {}
    for b, (k, choices) in zip(BASES, reversed(SEARCH_SPACE.items())):
        target = np.arange(len(choices)) if return_by_index else np.asarray(choices)
        result[k] = target[indices % b]
        indices //= b

    return pd.DataFrame(result)


def _validation(n_checks: int = 10 ** 6):
    for i, ps in enumerate(itertools.product(*(c for c in SEARCH_SPACE.values()))):
        ans = {k: ps[-i-1] for i, k in enumerate(reversed(SEARCH_SPACE.keys()))}
        pred = convert(i)
        n_checks -= 1

        if ans != pred:
            raise ValueError(i, ans, pred)
        if n_checks == 0:
            print("Passed the test")
            break


if __name__ == "__main__":
    _validation()
