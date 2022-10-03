## f-ANOVA for local space

This library aims to figure out which parameters are important depending on the phase of optimizations.

This repository is based on the paper `Revisiting Hyperparameter Importance Assessment: Local Importance from the Lebesgue View`.


## Motivation

The key insight of our method is the fact that hyperparameter importance (HPI) depends on the definition of search space, but it has not been investigated how to define local search space.

For example, we use [NMT-Bench](https://github.com/Este1le/hpo_nmt) as a target example.
We first calculate the global HPI ($\gamma = 0.1$) using our method and then obtain the following result (We normalized the HPI so that the sum of HPI becomes 1).
According to this result, the HPI of `n_heads` is dominant as it takes up 96.7% of importance.

<img src="figs/nmt-global-hpi.png" alt="">

On the other hand, if we limit the search space to the top-10% domain,
we obtain the following HPI ($\gamma^\prime = 0.01$):

<img src="figs/nmt-local-hpi.png" alt="">

Furthermore, the ratios of HPIs for each calculation are the following:

<img src="figs/nmt-hpi-ratio.png" alt="">

It just implies that even though we need to tune intensively tune `n_heads` to obtain the top-10% performance, the most important hyperparameter for the top-1% is not `n_heads`, but `bpe`.

To better-tune hyperparameters, we need to gain insight into which hyperparameters to focus at each phase of optimizations.
Our method tackles such an issue.

## Install

First, install the dependency.

```shell
$ pip install -r requirements.txt
```

Run this test:

```shell
$ python example.py
```

## Citation

Please cite the following paper (later):
Paper under double-blind review.
