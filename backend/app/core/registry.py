"""Model registry -- explicit manifest, never directory scanning.

Registers only the four conformance references: the classifier reference
in backend/models/example_logistic_regression/, and the clusterer,
regressor, and dimensionality_reducer references in
backend/tests/reference_models/.

Group submissions under backend/models/group_*/ are never imported here
until explicitly instructed (see CLAUDE.md "Current state").
"""
import logging

from models.base_model import BaseModel
from models.example_logistic_regression.model import LogisticRegressionModel
from tests.reference_models.ref_kmeans import RefKMeansModel
from tests.reference_models.ref_linear import RefLinearRegressionModel
from tests.reference_models.ref_pca import RefPCAModel

logger = logging.getLogger(__name__)

# One import, one manifest entry, per model.
MODEL_MANIFEST: dict[str, type] = {
    "logistic_regression": LogisticRegressionModel,
    "kmeans": RefKMeansModel,
    "linear_regression": RefLinearRegressionModel,
    "pca": RefPCAModel,
}


def build_registry(manifest: dict[str, type] = MODEL_MANIFEST) -> dict[str, type[BaseModel]]:
    """Validate the manifest and return the entries that pass.

    Confirms each entry subclasses BaseModel. A bad entry is logged and
    omitted, never raised -- one broken submission must not take the
    whole registry down.
    """
    registry: dict[str, type[BaseModel]] = {}
    for name, cls in manifest.items():
        if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
            logger.error(
                "Registry entry '%s' (%r) does not subclass BaseModel; "
                "omitting.", name, cls,
            )
            continue
        registry[name] = cls
    return registry


REGISTRY: dict[str, type[BaseModel]] = build_registry()
