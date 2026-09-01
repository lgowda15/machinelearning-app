"""Model registry -- explicit manifest, never directory scanning.

Registers the conformance references (classifier, clusterer, regressor) plus
the group submissions that have been explicitly instructed for registration
(see CLAUDE.md "Current state" and GROUP_REMEDIATION.md). Group submissions
under backend/models/group_*/ are never imported here until so instructed.

group_03_rnn was the first group submission registered. It ships three
regressor architectures -- RNNModel, LSTMModel, GRUModel -- rather than the
single model.py entry point other groups use; all three are registered here
as separate manifest entries.

group_01_decision_trees ships four classifier algorithms (CART, CHAID, ID3,
oblique) in one folder, each registered under its own plain algorithm-name
key -- same convention as group_03's rnn/lstm/gru entries.

group_15_pca supplies a real PCA implementation, so it takes over the "pca"
key from RefPCAModel (the dimensionality_reducer conformance reference).
RefPCAModel's file is untouched in backend/tests/reference_models/ and can
still be imported directly by anything that wants the reference
specifically -- it's just no longer in the live manifest.

group_11_lda_qda ships two classifier algorithms (LDA, QDA) in one folder,
each registered under its own plain algorithm-name key -- same convention
as group_01 and group_03. Its folder also exposes analyze_suitability and
compare_lda_qda helpers; those aren't part of the BaseModel contract and
are not registered here.

group_06_regression ships one RegressionModel that adapts its strategy to
input shape (a stationarity-driven differencing heuristic for a single
feature, polynomial expansion for two or more), materially different
behaviour from RefLinearRegressionModel's plain OLS fit -- registered
under its own "regression" key rather than taking over an existing one,
the same way rnn/lstm/gru are distinct regressor techniques from each
other. RefLinearRegressionModel's old "linear_regression" key was later
removed from the live manifest -- see the note below.

group_05_knn_kmeans_gmm ships three algorithms in one folder: KNNModel
(new "knn" key, classifier), KMeansModel, and GMMModel (new "gmm" key,
clusterer). KMeansModel is a real K-Means implementation of the same
algorithm RefKMeansModel stands in for, under the same clusterer
model_type -- same situation as group_15/PCA, so it takes over the
"kmeans" key from RefKMeansModel rather than getting a separate one.
RefKMeansModel's file is untouched in backend/tests/reference_models/
and can still be imported directly by anything that wants the reference
specifically -- it's just no longer in the live manifest.

group_02_random_forest_xgboost ships two classifier algorithms in one
folder -- RandomForestModel and XGBoostModel -- each registered under
its own plain algorithm-name key ("random_forest", "xgboost"), same
convention as group_01/group_03/group_11. Neither takes over a
conformance-reference key; there is no reference random forest or
XGBoost stand-in among the four reference models.

group_09_dbscan_hierarchical ships two clusterer algorithms in one
folder -- DBSCANModel ("dbscan") and HierarchicalClusteringModel
("hierarchical_clustering") -- same plain-algorithm-name convention as
group_01/group_02/group_03/group_11. Neither takes over a
conformance-reference key; RefKMeansModel's old "kmeans" slot already
went to group_05, and there is no reference DBSCAN or hierarchical
stand-in among the four reference models. Neither has a native
predict() on unseen data -- see DECISIONS.md Session E for the accepted
resolution (exact-match fast path, then a deterministic nearest-
neighbor-style fallback).

group_07_cnn ships one CNNModel (new "cnn" key, classifier) -- a small
two-conv-layer network for flattened 28x28-grayscale-image input
(EXPECTED_N_FEATURES = 784, per CODING_STANDARDS.md SS4's note on
sequence/image groups). No existing reference to take over; no reference
CNN stand-in exists among the four reference models.

RefLinearRegressionModel's "linear_regression" key was removed from the
live manifest at the user's explicit instruction, once group_06's
"regression" key gave the regressor model_type real, actively-registered
coverage of its own -- see DECISIONS.md for the full record. Its file is
untouched in backend/tests/reference_models/ and can still be imported
directly by anything that wants the reference specifically (e.g. the
conformance suite's four-reference baseline check); it's just no longer
in MODEL_MANIFEST, same as RefPCAModel and RefKMeansModel above.
"""
import logging

from models.base_model import BaseModel
from models.example_logistic_regression.model import LogisticRegressionModel
from models.group_01_decision_trees.cart import CARTModel
from models.group_01_decision_trees.chaid import CHAIDModel
from models.group_01_decision_trees.id3 import ID3Model
from models.group_01_decision_trees.oblique_tree import ObliqueDecisionTreeModel
from models.group_02_random_forest_xgboost.random_forest import RandomForestModel
from models.group_02_random_forest_xgboost.xgboost_model import XGBoostModel
from models.group_03_rnn.gru import GRUModel
from models.group_03_rnn.lstm import LSTMModel
from models.group_03_rnn.rnn import RNNModel
from models.group_05_knn_kmeans_gmm.gmm import GMMModel
from models.group_05_knn_kmeans_gmm.kmeans import KMeansModel
from models.group_05_knn_kmeans_gmm.knn import KNNModel
from models.group_06_regression.model import RegressionModel
from models.group_07_cnn.model import CNNModel
from models.group_09_dbscan_hierarchical.dbscan import DBSCANModel
from models.group_09_dbscan_hierarchical.hierarchical import (
    HierarchicalClusteringModel,
)
from models.group_11_lda_qda.lda import LDAModel
from models.group_11_lda_qda.qda import QDAModel
from models.group_13_svm.model import SVMModel
from models.group_15_pca.model import PCAModel

logger = logging.getLogger(__name__)

# One import, one manifest entry, per model.
MODEL_MANIFEST: dict[str, type] = {
    "logistic_regression": LogisticRegressionModel,
    "kmeans": KMeansModel,
    "pca": PCAModel,
    "rnn": RNNModel,
    "lstm": LSTMModel,
    "gru": GRUModel,
    "cart": CARTModel,
    "chaid": CHAIDModel,
    "id3": ID3Model,
    "oblique_tree": ObliqueDecisionTreeModel,
    "svm": SVMModel,
    "lda": LDAModel,
    "qda": QDAModel,
    "regression": RegressionModel,
    "knn": KNNModel,
    "gmm": GMMModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "dbscan": DBSCANModel,
    "hierarchical_clustering": HierarchicalClusteringModel,
    "cnn": CNNModel,
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
