## PED-ANOVA

This repository is based on the paper `PED-ANOVA: Efficiently Quantifying Hyperparameter Importance in Arbitrary Subspaces`.

## Setup

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
