"""Group 01 -- Decision Trees.

Four independently selectable decision-tree classifiers, one class per
algorithm and one file per class, all subclassing
:class:`models.base_model.BaseModel`:

* :class:`~models.group_01_decision_trees.id3.ID3Model` -- entropy /
  information gain, multi-way splits, attribute used once per path.
* :class:`~models.group_01_decision_trees.cart.CARTModel` -- Gini, binary
  splits, cost-complexity pruning.
* :class:`~models.group_01_decision_trees.chaid.CHAIDModel` -- chi-square
  significance testing with adjacent-category merging, multi-way splits.
* :class:`~models.group_01_decision_trees.oblique_tree.ObliqueDecisionTreeModel`
  -- sparse oblique (multivariate hyperplane) splits; the modern technique
  for this group.
"""

from .cart import CARTModel
from .chaid import CHAIDModel
from .id3 import ID3Model
from .oblique_tree import ObliqueDecisionTreeModel

__all__ = [
    "CARTModel",
    "CHAIDModel",
    "ID3Model",
    "ObliqueDecisionTreeModel",
]
