import time

import numpy as np
import torch
from torch import nn

from models.base_model import BaseModel


class RNNModel(BaseModel):
    """Vanilla RNN regressor for sequence-style forecasting from tabular input.

    Group 3 fixed sequence contract:
    - Input X must be 2D float64 with exactly 6 columns.
    - Flattened feature order per row:
      [t0_f0, t0_f1, t1_f0, t1_f1, t2_f0, t2_f1]
    - Internal reshape: (n_samples, 6) -> (n_samples, 3, 2).
    """

    EXPECTED_N_FEATURES = 6
    EXPECTED_LOOKBACK = 3
    EXPECTED_FEATURES_PER_STEP = 2

    def __init__(
        self,
        lookback: int = 3,
        hidden_size: int = 16,
        num_layers: int = 1,
        learning_rate: float = 1e-2,
        max_epochs: int = 120,
        patience: int = 12,
        min_delta: float = 1e-6,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            lookback=lookback,
            hidden_size=hidden_size,
            num_layers=num_layers,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            random_state=random_state,
            **kwargs,
        )
        if lookback <= 0:
            raise ValueError("lookback must be a positive integer.")
        if lookback != self.EXPECTED_LOOKBACK:
            raise ValueError(
                f"lookback must be {self.EXPECTED_LOOKBACK} for Group 3 schema."
            )
        if hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer.")
        if num_layers <= 0:
            raise ValueError("num_layers must be a positive integer.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be a positive integer.")
        if patience <= 0:
            raise ValueError("patience must be a positive integer.")

        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.random_state = random_state
        self._train_time = None
        self._final_loss = None
        self._features_per_step = None
        self._model = None
        self._device = torch.device("cpu")

    def _reshape_input(self, X: np.ndarray) -> np.ndarray:
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

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "RNNModel":
        if y is None:
            raise ValueError("RNNModel is supervised; y must not be None.")
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy ndarray.")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy ndarray.")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}.")
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError(f"y must be numeric, got dtype {y.dtype}.")
        if not np.isfinite(y).all():
            raise ValueError("y contains non-finite values (NaN or inf).")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows, y has {y.shape[0]}.")

        X_seq = self._reshape_input(X)
        self._features_per_step = X_seq.shape[2]

        torch.manual_seed(self.random_state)
        rnn_layer = nn.RNN(
            input_size=self._features_per_step,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            nonlinearity="tanh",
            batch_first=True,
        )
        head = nn.Linear(self.hidden_size, 1)
        rnn_layer.to(self._device)
        head.to(self._device)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(
            list(rnn_layer.parameters()) + list(head.parameters()),
            lr=self.learning_rate,
        )

        X_tensor = torch.tensor(X_seq, dtype=torch.float32, device=self._device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self._device).view(-1, 1)

        t0 = time.perf_counter()
        best_loss = float("inf")
        best_state = None
        epochs_without_improve = 0

        for _ in range(self.max_epochs):
            rnn_layer.train()
            head.train()
            optimizer.zero_grad(set_to_none=True)

            output, _ = rnn_layer(X_tensor)
            preds = head(output[:, -1, :])
            loss = loss_fn(preds, y_tensor)
            loss.backward()
            optimizer.step()

            current_loss = float(loss.detach().cpu().item())
            if current_loss + self.min_delta < best_loss:
                best_loss = current_loss
                best_state = {
                    "rnn": {
                        k: v.detach().cpu().clone()
                        for k, v in rnn_layer.state_dict().items()
                    },
                    "head": {
                        k: v.detach().cpu().clone()
                        for k, v in head.state_dict().items()
                    },
                }
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= self.patience:
                    break

        if best_state is not None:
            rnn_layer.load_state_dict(best_state["rnn"])
            head.load_state_dict(best_state["head"])

        self._model = {"rnn": rnn_layer, "head": head}
        self._train_time = time.perf_counter() - t0
        self._final_loss = best_loss

        self.is_fitted = True
        self.n_features = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy ndarray.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")
        if X.dtype != np.float64:
            raise ValueError(f"X must be float64, got {X.dtype}.")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values (NaN or inf).")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Model trained on {self.n_features} features, got {X.shape[1]}."
            )
        if self._model is None:
            raise RuntimeError("Internal model state is unavailable; call fit() again.")
        X_seq = self._reshape_input(X)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32, device=self._device)

        rnn_layer = self._model["rnn"]
        head = self._model["head"]
        rnn_layer.eval()
        head.eval()
        with torch.no_grad():
            output, _ = rnn_layer(X_tensor)
            preds = head(output[:, -1, :]).squeeze(1)
        return preds.detach().cpu().numpy().astype(float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        # Regressor: must return None
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        return None

    def get_metadata(self) -> dict:
        return {
            "model_name": "Vanilla RNN Regressor",
            "model_type": "regressor",
            "hyperparameters": self.hyperparams,
            "training_time_seconds": self._train_time,
            "n_features": self.n_features,
            "feature_importance": None,
        }
