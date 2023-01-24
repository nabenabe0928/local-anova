from typing import Any, Dict
import itertools

import numpy as np

import pandas as pd

from utils.constants import SEARCH_SPACE


BASES = [len(c) for c in reversed(SEARCH_SPACE.values())]


def convert(index: int, return_by_index: bool = False) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for b, (k, choices) in zip(BASES, reversed(SEARCH_SPACE.items())):
        target = np.arange(len(choices)) if return_by_index else choices
        result[k] = target[index % b]
        index //= b

    return result


def convert_multiple(
    indices: np.ndarray, return_by_index: bool = False
) -> pd.DataFrame:
    _indices = indices.copy()
    result: Dict[str, Any] = {}
    for b, (k, choices) in zip(BASES, reversed(SEARCH_SPACE.items())):
        target = np.arange(len(choices)) if return_by_index else np.asarray(choices)
        result[k] = target[_indices % b]
        _indices //= b

    return pd.DataFrame(result)


def _validation(n_checks: int = 10**6):
    for i, ps in enumerate(itertools.product(*(c for c in SEARCH_SPACE.values()))):
        ans = {k: ps[-i - 1] for i, k in enumerate(reversed(SEARCH_SPACE.keys()))}
        pred = convert(i)
        n_checks -= 1

        if ans != pred:
            raise ValueError(i, ans, pred)
        if n_checks == 0:
            print("Passed the test")
            break


if __name__ == "__main__":
    _validation()
