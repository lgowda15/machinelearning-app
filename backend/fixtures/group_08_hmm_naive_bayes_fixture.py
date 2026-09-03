"""Custom validate_submission.py fixture for group_08_hmm_naive_bayes.

validate_submission.py's generic fixture is a fixed (n_samples, 6) tabular
array (see make_generic_fixture in validate_submission.py). NaiveBayesModel
is fine with that shape, but HMMModel hard-requires exactly 10 features (5
timesteps x 2 features/step, see hmm.py's EXPECTED_N_FEATURES) and raises
ValueError on anything else, per its own sequence contract. This mirrors
group_07_cnn_fixture.py's precedent for a folder whose generic-fixture
column count doesn't match one of its models' fixed schema.

load_custom_fixture() applies one fixture per folder to every discovered
class in it, so this same (n, 10) fixture is also used for
NaiveBayesModel -- harmless, since NaiveBayesModel has no column-count
requirement.

make_fixture(seed) -> (X, y, X_test) must match the shape HMMModel expects:
X as (n_samples, 10) float64, y as 1D int labels with >=2 classes
(classifier contract, and HMMModel additionally requires at least
n_components training sequences per class -- N_TRAIN=120 over 2 classes
gives ~60 each, comfortably above the default n_components=3), X_test in
the same (n_samples, 10) shape.
"""
import numpy as np

N_FEATURES = 10
N_TRAIN = 120
N_TEST = 30
N_CLASSES = 2


def make_fixture(seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N_TRAIN, N_FEATURES)).astype(np.float64)
    y = rng.integers(0, N_CLASSES, N_TRAIN)
    X_test = rng.standard_normal((N_TEST, N_FEATURES)).astype(np.float64)
    return X, y, X_test
