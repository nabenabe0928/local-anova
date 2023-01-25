from argparse import ArgumentParser
import itertools
import json

import numpy as np

import pandas as pd

from utils.api_wrapper import BenchmarkWrapper
from utils.constants import PARAM_NAMES, SEARCH_SPACE, TASK_NAMES


FIXED_CONFIGS = {p: [] for p in PARAM_NAMES}
FIXED_CONFIGS[PARAM_NAMES[0]] = 0.0
FIXED_CONFIGS[PARAM_NAMES[1]] = 0.0
for ps in itertools.product(*(SEARCH_SPACE[p] for p in PARAM_NAMES[2:])):
    for name, p in zip(PARAM_NAMES[2:], ps):
        FIXED_CONFIGS[name].append(p)

FIXED_CONFIGS["Optimizer"] = "SGD"
FIXED_CONFIGS["Resolution"] = 1.0


def save_results(dataset_name: str) -> None:
    bench = BenchmarkWrapper(task=dataset_name)
    config_table = pd.DataFrame(FIXED_CONFIGS)
    results = []

    for ps in itertools.product(
        *(
            SEARCH_SPACE["LearningRate"],
            SEARCH_SPACE["WeightDecay"],
        )
    ):
        print(ps)
        config_table[PARAM_NAMES[0]] = ps[0]
        config_table[PARAM_NAMES[1]] = ps[1]
        preds = bench(config_table)["valid-acc"]
        results.append(preds.to_numpy())

    with open(f"data/{dataset_name}.json", mode="w") as f:
        divisor = 1000
        rounded_data = (
            np.asarray(np.hstack(results) * divisor, dtype=np.int32) / divisor
        )
        print(rounded_data, rounded_data.min())
        json.dump(rounded_data.tolist(), f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=TASK_NAMES, default="cifar10")
    args = parser.parse_args()

    dataset_name = args.dataset
    save_results(dataset_name)
