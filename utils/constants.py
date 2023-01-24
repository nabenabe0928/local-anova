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
    Op6=(0, 1, 2, 3, 4),
)
CATEGORICALS = dict(
    LearningRate=False,
    WeightDecay=False,
    N=False,
    W=False,
    Activation=True,
    TrivialAugment=True,
    Op1=True,
    Op2=True,
    Op3=True,
    Op4=True,
    Op5=True,
    Op6=True,
)
PARAM_NAMES = list(SEARCH_SPACE.keys())
TASK_NAMES = ["cifar10", "fashion_mnist", "colorectal_histology"]
DATASET_DICT = {
    "cifar10": "CIFAR10",
    "fashion_mnist": "Fashion-MNIST",
    "colorectal_histology": "Colorectal-Histology",
}
