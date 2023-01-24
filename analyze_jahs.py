from argparse import ArgumentParser

import numpy as np
import ujson as json

from anova import analyze
from utils.constants import CATEGORICALS, SEARCH_SPACE, TASK_NAMES
from utils.index_to_config import convert_multiple


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=TASK_NAMES, default="cifar10")
    args = parser.parse_args()

    rng = np.random.RandomState(0)
    dataset_name = args.dataset
    F_all = np.asarray(json.load(open(f"data/{dataset_name}.json")))

    n_samples, N = 10**5, F_all.size
    indices = rng.choice(np.arange(N), size=n_samples, replace=False)
    X = convert_multiple(indices, return_by_index=True)

    F = F_all[indices]
    print(dataset_name)
    # accuracy, but not error rate, so we flip the sign
    result = analyze(X, -F, search_space=SEARCH_SPACE, gamma_local=0.1, categoricals=CATEGORICALS)
    for k, v in reversed(result.items()):
        print(f"{k}: {v * 100:.2f}%")
