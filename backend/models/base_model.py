from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np


class BaseModel(ABC):
    """Common interface for all models in the integration platform.

    The integration layer calls every model through these four methods,
    so it never needs to know which algorithm is inside.
    """

    def __init__(self, **hyperparams):
        self.hyperparams = hyperparams
        self.is_fitted = False
        self.n_features = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseModel":
        """Train the model and return self.

        X: 2D float64 array (n_samples, n_features), already preprocessed
           by the backend. y: 1D array (n_samples,); None for clustering
           and dimensionality-reduction models.
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return a 1D array of shape (n_samples,)."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Classifiers: (n_samples, n_classes) probabilities summing to 1,
        columns ordered to match self.classes_. Other model types: None.
        """
        ...

    @abstractmethod
    def get_metadata(self) -> Dict:
        """Return the metadata dict specified in the coding standards."""
        ...
