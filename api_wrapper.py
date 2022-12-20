from typing import Iterable, Literal, Optional
import os

import pandas as pd

from jahs_bench import Benchmark


# https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/v1.1.0/assembled_surrogates.tar
DATA_DIR = f"{os.environ['HOME']}/tabular_benchmarks/jahs_bench_data/"


class BenchmarkWrapper(Benchmark):
    def __init__(
        self,
        task: Literal["colorectal_histology", "cifar10", "fashion_mnist"] = "cifar10",
        download: bool = False,
        save_dir: str = DATA_DIR,
        metrics: Optional[Iterable[str]] = ["valid-acc"][:],
    ):
        super().__init__(task=task, download=download, save_dir=save_dir, metrics=metrics)

    def __call__(
        self,
        feats: pd.DataFrame,
        nepochs: Optional[int] = 200,
    ) -> pd.DataFrame:

        assert nepochs > 0
        feats.loc[:, "epoch"] = nepochs

        outputs = []
        for model in self._surrogates.values():
            outputs.append(model.predict(feats))

        return pd.concat(outputs, axis=1)
