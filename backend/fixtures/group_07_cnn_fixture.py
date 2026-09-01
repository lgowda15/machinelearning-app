"""Custom validate_submission.py fixture for group_07_cnn.

validate_submission.py's generic fixture is a fixed (n_samples, 6) tabular
array (see make_generic_fixture in validate_submission.py) -- fine for plain
tabular models, but CNNModel hard-requires exactly 784 features (a flattened
28x28 image) and raises ValueError on anything else per its own input
contract. validate_submission.py's module docstring anticipates exactly this
case for groups 3/7/8 and looks for this file by name; nothing existed here
yet (group_03_rnn's EXPECTED_N_FEATURES happens to be 6, matching the
generic default by coincidence, so it never surfaced this gap).

make_fixture(seed) -> (X, y, X_test) must match the shape CNNModel expects:
X as (n_samples, 784) float, y as 1D int labels with >=2 classes (classifier
contract), X_test in the same (n_samples, 784) shape.
"""
import numpy as np

N_FEATURES = 784
N_TRAIN = 120
N_TEST = 30
N_CLASSES = 3


def make_fixture(seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N_TRAIN, N_FEATURES))
    y = rng.integers(0, N_CLASSES, N_TRAIN)
    X_test = rng.standard_normal((N_TEST, N_FEATURES))
    return X, y, X_test
