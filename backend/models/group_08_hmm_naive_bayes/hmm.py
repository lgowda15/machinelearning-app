import time

import numpy as np
from hmmlearn.hmm import GaussianHMM

from models.base_model import BaseModel


class HMMModel(BaseModel):
    """Hidden Markov Model classifier (generative, one HMM per class).

    Group 8 fixed sequence contract (CODING_STANDARDS.md Section 4 -- HMM
    is a sequence model, so it receives the same 2D (n_samples,
    n_features) input as every other model and reshapes internally):
    - Input X must be 2D float64 with exactly 10 columns.
    - Flattened feature order per row:
      [t0_f0, t0_f1, t1_f0, t1_f1, t2_f0, t2_f1, t3_f0, t3_f1, t4_f0, t4_f1]
    - Internal reshape: (n_samples, 10) -> (n_samples, 5, 2).

    Classification approach: a Gaussian HMM is fit per class on that
    class's training sequences (Baum-Welch). A new sequence is scored
    under every class's HMM and assigned the class whose HMM gives it the
    highest log-likelihood -- the standard generative pattern for
    sequence classification with HMMs, distinct from using a single HMM's
    hidden states as clusters.
    """

    EXPECTED_LOOKBACK = 5
    EXPECTED_FEATURES_PER_STEP = 2
    EXPECTED_N_FEATURES = EXPECTED_LOOKBACK * EXPECTED_FEATURES_PER_STEP

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "diag",
        n_iter: int = 50,
        lookback: int = 5,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            lookback=lookback,
            random_state=random_state,
            **kwargs,
        )
        if n_components <= 0:
            raise ValueError("n_components must be a positive integer.")
        if covariance_type not in ("diag", "full", "tied", "spherical"):
            raise ValueError(
                f"covariance_type must be one of "
                f"'diag', 'full', 'tied', 'spherical'; got {covariance_type!r}."
            )
        if n_iter <= 0:
            raise ValueError("n_iter must be a positive integer.")
        if lookback != self.EXPECTED_LOOKBACK:
            raise ValueError(
                f"lookback must be {self.EXPECTED_LOOKBACK} for Group 8 schema."
            )

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.lookback = lookback
        self.random_state = random_state
        self._train_time = None
        self._models: dict = {}
        self.classes_ = None

    def _reshape_input(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy ndarray.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")
        if X.dtype != np.float64:
            raise ValueError(f"X must be float64, got {X.dtype}.")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values (NaN or inf).")
        if X.shape[1] != self.EXPECTED_N_FEATURES:
            raise ValueError(
                f"X must have exactly {self.EXPECTED_N_FEATURES} features, "
                f"got {X.shape[1]}."
            )
        return X.reshape(
            X.shape[0],
            self.EXPECTED_LOOKBACK,
            self.EXPECTED_FEATURES_PER_STEP,
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "HMMModel":
        if y is None:
            raise ValueError("HMMModel is supervised; y must not be None.")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy ndarray.")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}.")
        X_seq = self._reshape_input(X)
        if X_seq.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X_seq.shape[0]} rows, y has {y.shape[0]}.")

        classes = np.unique(y)
        if classes.shape[0] < 2:
            raise ValueError(
                "HMMModel requires at least 2 classes in y to compare "
                "per-class HMM likelihoods against one another."
            )

        t0 = time.perf_counter()
        models = {}
        for cls in classes:
            rows = X_seq[y == cls]
            if rows.shape[0] < self.n_components:
                raise ValueError(
                    f"Class {cls!r} has {rows.shape[0]} training sequences, "
                    f"fewer than n_components={self.n_components}. HMM "
                    "cannot reliably estimate that many hidden states from "
                    "fewer sequences than states; lower n_components or "
                    "supply more data for this class."
                )
            observations = rows.reshape(-1, self.EXPECTED_FEATURES_PER_STEP)
            lengths = [self.EXPECTED_LOOKBACK] * rows.shape[0]
            hmm = GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            hmm.fit(observations, lengths=lengths)
            models[cls] = hmm
        self._train_time = time.perf_counter() - t0

        self._models = models
        self.classes_ = classes
        self.is_fitted = True
        self.n_features = X.shape[1]
        return self

    def _class_log_likelihoods(self, X: np.ndarray) -> np.ndarray:
        """(n_samples, n_classes) log-likelihood of each sequence under
        each class's HMM, columns ordered to match self.classes_."""
        X_seq = self._reshape_input(X)
        scores = np.column_stack([
            [self._models[cls].score(seq) for seq in X_seq]
            for cls in self.classes_
        ])
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Model trained on {self.n_features} features, got {X.shape[1]}."
            )
        scores = self._class_log_likelihoods(X)
        best = np.argmax(scores, axis=1)
        return self.classes_[best]

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Softmax of per-class log-likelihoods.

        Under a uniform class prior, P(class | X) is proportional to the
        per-class likelihood P(X | class). Normalising the log-likelihoods
        with a softmax is exactly that Bayes posterior, not an
        approximation of it -- softmax(log p_1, ..., log p_k) =
        p_i / sum_j(p_j) for every i.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Model trained on {self.n_features} features, got {X.shape[1]}."
            )
        scores = self._class_log_likelihoods(X)
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def get_metadata(self) -> dict:
        return {
            "model_name": "Hidden Markov Model Classifier",
            "model_type": "classifier",
            "hyperparameters": self.hyperparams,
            "training_time_seconds": self._train_time,
            "n_features": self.n_features,
            # Columns are a flattened (timestep, feature) sequence, not
            # independent tabular features -- a per-column score has no
            # coherent interpretation once reshaped, so None rather than
            # a misleading number per flattened column.
            "feature_importance": None,
        }
