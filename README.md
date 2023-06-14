## PED-ANOVA

This repository is based on the paper `PED-ANOVA: Efficiently Quantifying Hyperparameter Importance in Arbitrary Subspaces`.
Our method provides an easy-to-implement new f-ANOVA algorithm.

## Simple example using Sci-Kit Learn

As mentioned in the paper, our method is very easy to implement.
Here is an implementation example using Sci-Kit Learn.

```python
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


# Objective function for analysis
def sphere(x: np.ndarray) -> np.ndarray:
    coef = np.array([1, 5])
    x *= coef
    return np.sum(x ** 2, axis=-1)


# Define the search space and collect observations using random search
lb, ub = -5, 5
n_samples, dim = 1000, 2
x = np.random.random((n_samples, dim)) * (ub - lb) + lb
y = sphere(x)
order = np.argsort(y)
x_sorted = x[order]

# Define the top quantile and compute the local/global PDFs for each dimension
# Note that we are using Eq. (14) to calculate the KDEs rather tahn Eq. (15), which is a quicker version
quantile = 0.1
local_kdes = {
    f"x{d}": KernelDensity(kernel="gaussian").fit(x_sorted[:int(quantile * n_samples), d][:, None])
    for d in range(dim)
}
global_kdes = {
    f"x{d}": KernelDensity(kernel="gaussian").fit(x_sorted[:, d][:, None])
    for d in range(dim)
}

# Compute the global HPI of each dimension based on Eq. (16)
dx = np.linspace(lb, ub, 100)[:, None]
hpi_dict = {}
hpi_total = 0.0
for d in range(dim):
    pdf_global = np.exp(global_kdes[f"x{d}"].score_samples(dx))
    pdf_local = np.exp(local_kdes[f"x{d}"].score_samples(dx))
    hpi = pdf_global @ ((pdf_local / pdf_global - 1) ** 2)
    hpi_dict[f"x{d}"] = hpi
    hpi_total += hpi
else:
    hpi_dict = {k: hpi / hpi_total for k, hpi in hpi_dict.items()}

# Print the global HPI
print(hpi_dict)
```

Note that this example does not use the discretization trick, so it would take much more time to compute the global HPI compared to what we can expect from the KDEs built by Eq. (15) if `n_samples` is a large number such as $10^9$.

## Setup for the experiments in our paper

Install the dependency.

```shell
$ pip install -r requirements.txt
```

Run this test:

```shell
$ python analyze_toy.py
```

## Analysis of JAHS-Bench-201

1. Download [assembled_surrogates.tar](https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/v1.1.0/assembled_surrogates.tar)
2. Make a directory named `tabular_benchmarks/jahs_bench_data/` under the home directory
2. Locate the uncompressed `assembled_surrogates` under `~/tabular_benchmarks/jahs_bench_data/`

Then run the following:
```shell
./run_jahs.sh
```
# Citations

For the citation, use the following format:
```
@article{watanabe2023ped,
  title={{PED-ANOVA}: Efficiently Quantifying Hyperparameter Importance in Arbitrary Subspaces},
  author={S. Watanabe and A. Bansal and F. Hutter},
  journal={International Joint Conference on Artificial Intelligence},
  year={2023}
}
```
