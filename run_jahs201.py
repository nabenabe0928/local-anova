"""
Configuration space object:
  Hyperparameters:
    LearningRate, Type: UniformFloat, Range: [0.001, 1.0], Default: 0.1, on log-scale
    WeightDecay, Type: UniformFloat, Range: [1e-05, 0.01], Default: 0.0005, on log-scale
    N, Type: Ordinal, Sequence: {1, 3, 5}, Default: 1
    W, Type: Ordinal, Sequence: {4, 8, 16}, Default: 4
    Activation, Type: Categorical, Choices: {ReLU, Hardswish, Mish}, Default: ReLU
    TrivialAugment, Type: Categorical, Choices: {True, False}, Default: False
    Op1, Type: Categorical, Choices: {0, 1, 2, 3, 4}, Default: 0
    Op2, Type: Categorical, Choices: {0, 1, 2, 3, 4}, Default: 0
    Op3, Type: Categorical, Choices: {0, 1, 2, 3, 4}, Default: 0
    Op4, Type: Categorical, Choices: {0, 1, 2, 3, 4}, Default: 0
    Op5, Type: Categorical, Choices: {0, 1, 2, 3, 4}, Default: 0
    Op6, Type: Categorical, Choices: {0, 1, 2, 3, 4}, Default: 0
    * (fixed) Optimizer, Type: Categorical, Choices: {SGD}, Default: SGD
    * (budget) Resolution, Type: Ordinal, Sequence: {0.25, 0.5, 1.0}, Default: 1.0

n_configs = c x c x 3 x 3 x 3 x 2 x 5^6
          = c x c x 843750
          = 84375000
          < 10^8
"""
import os

import jahs_bench


DATA_DIR = f"{os.environ['HOME']}/tabular_benchmarks/jahs_bench_data/"


tasks = ["colorectal_histology", "cifar10", "fashion_mnist"]
benchmark = jahs_bench.Benchmark(task="cifar10", download=False, save_dir=DATA_DIR)

config = benchmark.sample_config(random_state=42)
results = benchmark(config, nepochs=200)

print(config)
print(results)
