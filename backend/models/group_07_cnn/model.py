"""Convolutional Neural Network model for the ML Integration Platform."""

import time

import numpy as np
import torch
from torch import nn

from models.base_model import BaseModel

# Required for deterministic behaviour.
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)


class _CNNNetwork(nn.Module):
    """Small CPU-friendly convolutional neural network."""

    def __init__(self, n_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


class CNNModel(BaseModel):
    """Convolutional Neural Network classifier."""

    EXPECTED_N_FEATURES = 784

    def __init__(
        self,
        epochs: int = 10,
        learning_rate: float = 0.001,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            epochs=epochs,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs,
        )

        if epochs <= 0:
            raise ValueError("epochs must be greater than zero.")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

        self._model: _CNNNetwork | None = None
        self.classes_: np.ndarray | None = None
        self._train_time: float | None = None

    def _validate_X(self, X: np.ndarray) -> None:
        """Validate the common input contract."""
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy ndarray.")

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")

        if X.shape[1] != self.EXPECTED_N_FEATURES:
            raise ValueError(
                "CNNModel expects "
                f"{self.EXPECTED_N_FEATURES} image features, "
                f"got {X.shape[1]}."
            )

        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values.")

    def _reshape_images(self, X: np.ndarray) -> torch.Tensor:
        """Convert flattened images to NCHW tensors."""
        side = int(np.sqrt(X.shape[1]))

        if side * side != X.shape[1]:
            raise ValueError(
                "CNNModel requires a square image layout; "
                f"received {X.shape[1]} features."
            )

        images = X.reshape(X.shape[0], 1, side, side)

        return torch.tensor(images, dtype=torch.float32)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> "CNNModel":
        """Train the CNN and return this model."""
        self._validate_X(X)

        if y is None:
            raise ValueError("CNNModel is supervised; y must not be None.")

        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy ndarray.")

        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}.")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows, "
                f"y has {y.shape[0]} rows."
            )

        if y.shape[0] == 0:
            raise ValueError("y must contain at least one sample.")

        if not np.isfinite(y).all():
            raise ValueError("y contains non-finite values.")

        classes = np.unique(y)

        if classes.size < 2:
            raise ValueError(
                "CNNModel requires at least two classes for classification."
            )

        self.classes_ = classes

        # Re-seed before constructing the network so repeated fits
        # produce identical initial weights.
        torch.manual_seed(self.random_state)

        self._model = _CNNNetwork(len(classes))

        X_tensor = self._reshape_images(X)

        class_indices = np.searchsorted(classes, y)
        y_tensor = torch.tensor(class_indices, dtype=torch.long)

        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
        )

        criterion = nn.CrossEntropyLoss()

        self._model.train()

        start = time.perf_counter()

        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)

            logits = self._model(X_tensor)
            loss = criterion(logits, y_tensor)

            loss.backward()
            optimizer.step()

        self._train_time = time.perf_counter() - start

        self.is_fitted = True
        self.n_features = X.shape[1]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")

        self._validate_X(X)

        if self._model is None or self.classes_ is None:
            raise RuntimeError("CNN model is not properly fitted.")

        X_tensor = self._reshape_images(X)

        self._model.eval()

        with torch.no_grad():
            logits = self._model(X_tensor)
            indices = torch.argmax(logits, dim=1).cpu().numpy()

        return self.classes_[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")

        self._validate_X(X)

        if self._model is None:
            raise RuntimeError("CNN model is not properly fitted.")

        X_tensor = self._reshape_images(X)

        self._model.eval()

        with torch.no_grad():
            logits = self._model(X_tensor)
            probabilities = torch.softmax(logits, dim=1)

        return probabilities.cpu().numpy()

    def get_metadata(self) -> dict:
        """Return the platform-required metadata."""
        return {
            "model_name": "Convolutional Neural Network",
            "model_type": "classifier",
            "hyperparameters": self.hyperparams,
            "training_time_seconds": self._train_time,
            "n_features": self.n_features,
            "feature_importance": None,
        }
