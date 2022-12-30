import itertools
import json

import numpy as np

import pandas as pd

from api_wrapper import BenchmarkWrapper


SEARCH_SPACE = dict(
    LearningRate=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0),
    WeightDecay=(0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01),
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
TASK_NAME = ["cifar10", "fashion_mnist", "colorectal_histology"][2]

FIXED_CONFIGS = {p: [] for p in PARAM_NAMES}
FIXED_CONFIGS[PARAM_NAMES[0]] = 0.0
FIXED_CONFIGS[PARAM_NAMES[1]] = 0.0
for ps in itertools.product(*(SEARCH_SPACE[p] for p in PARAM_NAMES[2:])):
    for name, p in zip(PARAM_NAMES[2:], ps):
        FIXED_CONFIGS[name].append(p)

FIXED_CONFIGS["Optimizer"] = "SGD"
FIXED_CONFIGS["Resolution"] = 1.0
bench = BenchmarkWrapper(task=TASK_NAME)
config_table = pd.DataFrame(FIXED_CONFIGS)
results = []

for ps in itertools.product(*(
    SEARCH_SPACE["LearningRate"],
    SEARCH_SPACE["WeightDecay"],
)):
    print(ps)
    config_table[PARAM_NAMES[0]] = ps[0]
    config_table[PARAM_NAMES[1]] = ps[1]
    preds = bench(config_table)["valid-acc"]
    results.append(preds.to_numpy())

with open(f"{TASK_NAME}.json", mode="w") as f:
    divisor = 1000
    rounded_data = np.asarray(np.hstack(results) * divisor, dtype=np.int32) / divisor
    print(rounded_data, rounded_data.min())
    json.dump(rounded_data.tolist(), f, indent=4)
