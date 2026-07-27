from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Common interface for all models in the integration platform.

    The integration layer calls every model through these methods, so it
    never needs to know which algorithm is inside.
    """

    def __init__(self, **hyperparams):
        self.hyperparams = hyperparams
        self.is_fitted = False
        self.n_features = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "BaseModel":
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
    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Classifiers: (n_samples, n_classes) probabilities summing to 1,
        columns ordered to match self.classes_. Other model types: None.
        """
        ...

    @abstractmethod
    def get_metadata(self) -> dict:
        """Return the metadata dict specified in the coding standards."""
        ...

    def get_visualization_data(self) -> dict | None:
        """Optional: JSON-serialisable data for model-specific visuals.

        Override only if your model produces visuals the generic metrics
        cannot express (SHAP values, dendrogram linkage, explained
        variance, tree structure). Return raw numbers and arrays as
        lists/dicts — never figures, images, or file paths.
        """
        return None