# models/group_04_ann/model.py
"""
Artificial Neural Network (ANN) — Multi-Layer Perceptron
=========================================================
Supports classification and regression via a configurable fully-connected
feed-forward network built entirely in PyTorch.

The model receives preprocessed float64 input from the backend (scaled,
encoded, imputed) and never performs any preprocessing of its own.
"""

import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.base_model import BaseModel

# ── Determinism (required by §10) ────────────────────────────────────────────
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)


# ── Internal network definition ───────────────────────────────────────────────

def _build_network(
    in_features: int,
    hidden_sizes: List[int],
    out_features: int,
    activation: str,
    dropout_rate: float,
) -> nn.Sequential:
    """Return a fully-connected feed-forward network.

    Architecture
    ------------
    Input  →  [Linear → BatchNorm → Activation → Dropout] × L  →  Linear(out)

    BatchNorm before activation stabilises training across diverse datasets
    without requiring per-dataset learning-rate tuning.
    Dropout is applied after activation, not before, so the scale of the
    activations that BatchNorm sees is not distorted.
    """
    activation_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
    }
    if activation not in activation_map:
        raise ValueError(
            f"activation must be one of {list(activation_map)}, "
            f"got '{activation}'."
        )
    act_cls = activation_map[activation]

    layers: List[nn.Module] = []
    prev = in_features
    for h in hidden_sizes:
        layers += [
            nn.Linear(prev, h),
            nn.BatchNorm1d(h),
            act_cls(),
            nn.Dropout(p=dropout_rate),
        ]
        prev = h
    layers.append(nn.Linear(prev, out_features))
    return nn.Sequential(*layers)


# ── Public model class ────────────────────────────────────────────────────────

class ANNModel(BaseModel):
    """Artificial Neural Network (Multi-Layer Perceptron).

    Handles both classification (cross-entropy loss, softmax probabilities)
    and regression (MSE loss, scalar outputs) determined automatically from
    the dtype of ``y`` at fit time:

    * integer or boolean ``y``  → classifier
    * float ``y``               → regressor

    Parameters
    ----------
    hidden_sizes : list[int]
        Neuron counts for each hidden layer.  Default ``[128, 64]`` gives a
        two-hidden-layer network that generalises well on tabular data of
        moderate width without overfitting on small datasets.
    activation : str
        Non-linearity applied after each BatchNorm.  One of
        ``"relu"`` (default), ``"tanh"``, ``"leaky_relu"``, ``"elu"``.
    lr : float
        Adam learning rate.  Default ``1e-3`` is Adam's well-established
        default; reduce to ``1e-4`` on noisy regression targets.
    epochs : int
        Training epochs.  Default ``100`` fits within the 5-minute CPU
        ceiling for datasets up to ~50 k rows × 50 features.
    batch_size : int
        Mini-batch size.  Default ``64`` balances gradient-noise regularisation
        against throughput on CPU.
    dropout_rate : float
        Fraction of activations zeroed per forward pass during training.
        Default ``0.2``; set to ``0.0`` to disable.
    weight_decay : float
        L2 regularisation coefficient for Adam.  Default ``1e-4``.
    random_state : int
        Seed forwarded to torch for reproducibility.  Default ``42``.
    """

    # ── construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        hidden_sizes: List[int] = None,
        activation: str = "relu",
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 64,
        dropout_rate: float = 0.2,
        weight_decay: float = 1e-4,
        random_state: int = 42,
        **kwargs,
    ):
        # Mutable default must not be shared across instances.
        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        if not hidden_sizes or any(h <= 0 for h in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers.")
        if lr <= 0:
            raise ValueError("lr must be greater than 0.")
        if epochs <= 0:
            raise ValueError("epochs must be greater than 0.")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in the range [0, 1).")

        super().__init__(
            hidden_sizes=hidden_sizes,
            activation=activation,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            random_state=random_state,
            **kwargs,
        )

        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.random_state = random_state

        # Populated during fit().
        self._network: Optional[nn.Sequential] = None
        self._model_type: Optional[str] = None   # "classifier" | "regressor"
        self.classes_: Optional[np.ndarray] = None
        self._train_time: Optional[float] = None
        self._loss_history: List[float] = []

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _infer_task(y: np.ndarray) -> str:
        """Return 'classifier' if y holds discrete labels, else 'regressor'."""
        if np.issubdtype(y.dtype, np.integer) or np.issubdtype(y.dtype, np.bool_):
            return "classifier"
        # Float array with only whole-number values still treated as classifier
        # iff the unique count is small (≤ 20) and every value is an integer.
        unique = np.unique(y)
        if np.all(unique == unique.astype(int)) and len(unique) <= 20:
            return "classifier"
        return "regressor"

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        """float64 ndarray → float32 CPU tensor (PyTorch default is float32)."""
        return torch.from_numpy(X.astype(np.float32))

    # ── BaseModel interface ───────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ANNModel":
        """Train the network.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features), dtype float64
            Preprocessed feature matrix supplied by the backend.
        y : np.ndarray, shape (n_samples,)
            Target vector.  Must not be None.

        Returns
        -------
        self
        """
        # ── input validation ─────────────────────────────────────────────────
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy ndarray.")
        if y is None:
            raise ValueError(
                "ANNModel requires a target vector y; "
                "it is a supervised model and does not support clustering "
                "or dimensionality reduction."
            )
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a NumPy ndarray.")
        if X.ndim != 2:
            raise ValueError(
                f"X must be a 2-D array, got shape {X.shape}. "
                "The backend supplies (n_samples, n_features)."
            )
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but y has {y.shape[0]} elements; "
                "they must be the same length."
            )
        if X.shape[0] < 2:
            raise ValueError(
                f"Training requires at least 2 samples, got {X.shape[0]}."
            )

        # ── seed (per-fit, so fit() is reproducible regardless of call order) ─
        torch.manual_seed(self.random_state)

        # ── determine task ────────────────────────────────────────────────────
        self._model_type = self._infer_task(y)

        if self._model_type == "classifier":
            self.classes_ = np.unique(y)
            # Map class labels → contiguous 0-based indices.
            label_to_idx = {lbl: i for i, lbl in enumerate(self.classes_)}
            y_idx = np.array([label_to_idx[lbl] for lbl in y], dtype=np.int64)
            n_out = len(self.classes_)
            criterion: nn.Module = nn.CrossEntropyLoss()
            y_tensor = torch.from_numpy(y_idx)          # (n,) int64
        else:
            n_out = 1
            criterion = nn.MSELoss()
            y_tensor = self._to_tensor(y.reshape(-1, 1))  # (n, 1) float32

        # ── build network ─────────────────────────────────────────────────────
        self._network = _build_network(
            in_features=X.shape[1],
            hidden_sizes=self.hidden_sizes,
            out_features=n_out,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
        )
        self._network.train()

        # ── optimiser ─────────────────────────────────────────────────────────
        optimiser = torch.optim.Adam(
            self._network.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # ── data loader ───────────────────────────────────────────────────────
        X_tensor = self._to_tensor(X)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, X.shape[0]),
            shuffle=True,
            # Determinism: worker count 0 means data loading is in-process,
            # avoiding non-deterministic multi-process ordering.
            num_workers=0,
            drop_last=(
                X.shape[0] % min(self.batch_size, X.shape[0]) == 1
                and X.shape[0] > min(self.batch_size, X.shape[0])
            ),
        )

        # ── training loop ─────────────────────────────────────────────────────
        self._loss_history = []
        t0 = time.perf_counter()

        for _ in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in loader:
                optimiser.zero_grad()
                logits = self._network(X_batch)

                if self._model_type == "classifier":
                    loss = criterion(logits, y_batch)
                else:
                    loss = criterion(logits, y_batch)

                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
                n_batches += 1

            self._loss_history.append(epoch_loss / max(n_batches, 1))

        self._train_time = time.perf_counter() - t0

        # ── finalise ──────────────────────────────────────────────────────────
        self._network.eval()
        self.is_fitted = True
        self.n_features = X.shape[1]

        return self

    # ── inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions as a 1-D array of length n_samples.

        * Classifier → class labels (same dtype as training y).
        * Regressor  → continuous floats.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "This ANNModel instance has not been fitted yet. "
                "Call fit(X, y) before predict(X)."
            )
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy ndarray.")
        if X.ndim != 2:
            raise ValueError(
                f"X must be a 2-D array, got shape {X.shape}."
            )
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Model was trained on {self.n_features} features, "
                f"but X has {X.shape[1]} features."
            )

        X_tensor = self._to_tensor(X)
        with torch.no_grad():
            logits = self._network(X_tensor)

        if self._model_type == "classifier":
            idx = logits.argmax(dim=1).numpy()          # (n,) int
            return self.classes_[idx]                   # restore original labels
        else:
            return logits.squeeze(1).numpy().astype(np.float64)  # (n,)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Return class probabilities for classifiers; None for regressors.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes), rows summing to 1,
        columns ordered to match ``self.classes_``.
        None if this model is a regressor.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "This ANNModel instance has not been fitted yet. "
                "Call fit(X, y) before predict_proba(X)."
            )
        if self._model_type == "regressor":
            return None
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy ndarray.")
        if X.ndim != 2:
            return None

        if X.ndim != 2:
            raise ValueError(
                f"X must be a 2-D array, got shape {X.shape}."
            )
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Model was trained on {self.n_features} features, "
                f"but X has {X.shape[1]} features."
            )

        X_tensor = self._to_tensor(X)
        with torch.no_grad():
            logits = self._network(X_tensor)
            proba = torch.softmax(logits, dim=1).numpy()  # (n, n_classes)

        return proba.astype(np.float64)

    def get_metadata(self) -> Dict:
        """Return the standard metadata dict required by §8.

        ``feature_importance`` is not natively available for MLPs; the field
        is set to None rather than returning misleading scores.  If SHAP-based
        importance is needed, that is handled by a separate group (Group 2).
        """
        return {
            "model_name": "Artificial Neural Network (MLP)",
            "model_type": self._model_type if self.is_fitted else "classifier",
            "hyperparameters": self.hyperparams,
            "training_time_seconds": self._train_time,
            "n_features": self.n_features,
            "feature_importance": None,
        }

    # ── optional visualisation data ───────────────────────────────────────────

    def get_visualization_data(self) -> Optional[Dict]:
        """Return training-loss curve data for the frontend to render.

        The loss history lets the UI plot an epoch-vs-loss learning curve,
        which is the most informative per-run diagnostic for an ANN.

        Returns
        -------
        dict with keys:
            ``loss_history``  : list of float — mean batch loss per epoch.
            ``epochs``        : list of int   — 1-based epoch indices.
            ``model_type``    : str           — "classifier" or "regressor".
        """
        if not self.is_fitted:
            return None
        return {
            "loss_history": self._loss_history,
            "epochs": list(range(1, len(self._loss_history) + 1)),
            "model_type": self._model_type,
        }
