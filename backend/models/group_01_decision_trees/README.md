# Group 01 — Decision Trees

Four independently selectable decision-tree classifiers for the ML
integration platform. One class per algorithm, one file per class, all
subclassing `models.base_model.BaseModel` and all conforming to the input,
output, metadata and visualisation contracts in the Coding Standards
(v1.0).

```
backend/models/group_01_decision_trees/
├── __init__.py          # re-exports the four model classes
├── id3.py               # ID3Model
├── cart.py              # CARTModel
├── chaid.py             # CHAIDModel
├── oblique_tree.py      # ObliqueDecisionTreeModel  (the modern technique)
├── test.py              # unit tests (synthetic data only)
├── requirements.txt
└── README.md
```

---

## 1. Model

| Class | File | One-sentence description |
|---|---|---|
| `ID3Model` | `id3.py` | Quinlan's ID3: multi-way splits chosen by **entropy / information gain**, with each attribute used at most once per root-to-leaf path and no pruning. |
| `CARTModel` | `cart.py` | Breiman's CART: **binary** `feature <= threshold` splits chosen by the **Gini index**, with optional cost-complexity pruning (wraps `sklearn.tree.DecisionTreeClassifier`, which is an optimised CART). |
| `CHAIDModel` | `chaid.py` | Kass's CHAID: **multi-way** splits chosen by **Pearson chi-square significance** after adjacent-category merging, with Bonferroni-adjusted p-values. |
| `ObliqueDecisionTreeModel` | `oblique_tree.py` | Sparse **oblique** tree: each node tests a *linear combination* of features (`x·w + b <= t`) obtained from an L1-regularised logistic fit, falling back to an axis-parallel split when a hyperplane does not help. |

All four are `model_type: "classifier"`.

---

## 2. Usage

Every model follows the identical construct → fit → predict flow, so the
backend can treat them interchangeably.

```python
import numpy as np

from models.group_01_decision_trees import (
    CARTModel,
    CHAIDModel,
    ID3Model,
    ObliqueDecisionTreeModel,
)

# X_train / X_test: 2D float64, already scaled + encoded by the backend
# y_train: 1D array of class labels

model = CARTModel()                     # or ID3Model / CHAIDModel / ObliqueDecisionTreeModel
model.fit(X_train, y_train)             # returns self, so chaining also works

labels = model.predict(X_test)          # (n_samples,)
proba = model.predict_proba(X_test)     # (n_samples, n_classes), rows sum to 1
metadata = model.get_metadata()         # the six-key dict from Section 8
tree = model.get_visualization_data()   # {"tree_structure": {...}}, JSON-safe

# Column order of `proba` matches model.classes_
classes = model.classes_
```

Hyperparameters are passed to the constructor only:

```python
model = ID3Model(max_depth=6, n_bins=5, min_samples_leaf=10)
model = CHAIDModel(alpha_split=0.01, max_bins=8)
model = ObliqueDecisionTreeModel(oblique_C=0.1, criterion="entropy")
```

---

## 3. Hyperparameters

### `ID3Model`

| Parameter | Default | Purpose |
|---|---|---|
| `max_depth` | `3` | Maximum number of splits on any root-to-leaf path. ID3 splits multi-way and never prunes, so one level costs up to `n_bins`× as many nodes as a binary level — hence a shallower default than CART's. |
| `min_samples_split` | `20` | A node with fewer samples than this becomes a leaf. |
| `min_samples_leaf` | `1` | An attribute is rejected at a node if any child it would create is smaller than this. |
| `n_bins` | `4` | Maximum ordinal bins per feature — the size of the attribute alphabet ID3 sees. |
| `min_information_gain` | `0.0` | A node is split only when the best information gain strictly exceeds this, so zero-gain splits never occur. |
| `random_state` | `42` | Accepted for interface consistency and reported in metadata; the algorithm has no stochastic component, so it does not change the fitted tree. |

### `CARTModel`

| Parameter | Default | Purpose |
|---|---|---|
| `criterion` | `"gini"` | Split criterion: `"gini"` (the CART criterion), `"entropy"` or `"log_loss"`. |
| `max_depth` | `5` | Maximum depth of the tree. |
| `min_samples_split` | `2` | Minimum samples required at a node before it may be split. |
| `min_samples_leaf` | `1` | Minimum samples that must remain in each child. |
| `min_impurity_decrease` | `0.0` | A split is accepted only if it reduces weighted impurity by at least this much. |
| `ccp_alpha` | `0.0` | Cost-complexity pruning strength; `0.0` disables pruning, larger values prune harder. |
| `random_state` | `42` | Seed for tie-breaking between equally good splits. |

### `CHAIDModel`

| Parameter | Default | Purpose |
|---|---|---|
| `max_depth` | `4` | Maximum number of splits on any root-to-leaf path. CHAID splits multi-way, and on a large sample the chi-square test finds almost any association significant, so the depth cap is the practical guard on tree size. |
| `min_samples_split` | `20` | A node with fewer samples than this becomes a leaf (chi-square tests need reasonable cell counts). |
| `min_samples_leaf` | `10` | Merged category groups below this size are force-merged into a neighbour; a predictor that cannot satisfy it is rejected. |
| `max_bins` | `10` | Maximum number of node-local equal-frequency categories created for a continuous predictor before merging starts. |
| `alpha_merge` | `0.05` | Adjacent categories keep merging while the most similar neighbouring pair has a chi-square p-value above this. |
| `alpha_split` | `0.05` | A node is split only if the best Bonferroni-adjusted p-value is at most this. |
| `bonferroni` | `True` | Apply Kass's Bonferroni adjustment for the number of possible merges. |
| `random_state` | `42` | Accepted for interface consistency and reported in metadata; the algorithm has no stochastic component. |

### `ObliqueDecisionTreeModel`

| Parameter | Default | Purpose |
|---|---|---|
| `max_depth` | `5` | Maximum number of splits on any root-to-leaf path. |
| `min_samples_split` | `20` | A node with fewer samples than this becomes a leaf (a hyperplane needs enough samples to be meaningful). |
| `min_samples_leaf` | `5` | Minimum samples that must remain on each side of a split. |
| `criterion` | `"gini"` | Impurity used to score candidate thresholds: `"gini"` or `"entropy"`. |
| `oblique_C` | `1.0` | Inverse L1 regularisation strength of the per-node logistic fit; smaller values give sparser, more readable hyperplanes. |
| `max_hyperplane_iter` | `200` | Maximum solver iterations for the per-node logistic fit. |
| `min_impurity_decrease` | `0.0` | A split is accepted only if it reduces weighted impurity by more than this. |
| `random_state` | `42` | Seed passed to the per-node solver; fixes the fitted tree. |

---

## 4. Feature layout

No special layout is required and **no reshaping happens inside these
models**. They consume the standard platform input contract exactly as
delivered (Section 4 of the Coding Standards):

* `X` — `numpy.ndarray`, `dtype=float64`, 2D `(n_samples, n_features)`,
  already scaled, already numeric/encoded, no missing values,
* `y` — 1D array of class labels, length `n_samples`. Any label dtype
  works (integer, string, boolean); `predict` returns labels of the same
  dtype, and `predict_proba` columns are ordered to match `self.classes_`
  (`numpy.unique(y)`, i.e. ascending).

Column order is irrelevant to correctness — features are referred to
positionally as `feature_0 … feature_{n_features-1}` in
`feature_importance` and in the visualisation payload, so whatever column
order the backend supplies is what the frontend labels will describe.

The models perform **no** scaling, encoding, imputation, train/test
splitting, dataset loading or feature engineering, and never fit any
statistic on data passed to `predict`. Both `fit` and `predict` reject
non-finite input with a `ValueError` rather than silently imputing it.

Two algorithms need ordered *categories* rather than raw continuous
values, and create them from the training data inside the algorithm (see
Design decisions, §6):

* `ID3Model` discretises each column once, at `fit` time, into at most
  `n_bins` quantile bins computed from the training split only; the edges
  are stored on the model and reused unchanged by `predict`.
* `CHAIDModel` discretises **locally at each node**, from that node's own
  training samples; the edges and merged groups are stored on the node and
  form part of the split rule.

---

## 5. Running the tests

Run from the `backend/` directory (the platform root, the same directory
`validate_submission.py` is run from), so that `models.base_model`
resolves:

```bash
python -m pytest models/group_01_decision_trees/test.py --cov=models/group_01_decision_trees --cov-report=term-missing
```

Result on Python 3.11/3.12, CPU only: **184 tests pass, 98% statement
coverage** (every module ≥ 96%).

If you prefer to work from inside this folder, `test.py` puts the platform
root on `sys.path` itself (derived from `__file__`, no hardcoded paths), so
this also works:

```bash
python -m unittest test
```

All fixtures are synthetic numpy arrays generated with fixed seeds. No
dataset file is read, and none is committed.

---

## 6. Design decisions

### Why these four algorithms

ID3, CART and CHAID are the three algorithms allotted to Group 01. They are
implemented as three genuinely different algorithms, not three
configurations of one tree:

| | ID3 | CART | CHAID |
|---|---|---|---|
| Split selection | maximise **information gain** (entropy drop) | maximise **Gini** impurity decrease | minimise **Bonferroni-adjusted chi-square p-value** |
| Split arity | multi-way (one child per bin) | strictly binary | multi-way (one child per merged group) |
| Feature reuse on a path | never — the attribute is consumed | unrestricted | unrestricted |
| Stopping rule | depth / purity / no positive gain / attributes exhausted | depth / sample counts / impurity decrease | **statistical significance** (`alpha_split`) |
| Pruning | none | cost-complexity (`ccp_alpha`) | none (significance testing does the work) |
| Continuous features | ordinal bins fixed at fit time | native threshold search | ordinal bins created **per node**, then merged |

**How ID3 differs from CART.** ID3 uses entropy and information gain,
splits a node into as many branches as the attribute has values, and
removes the attribute from consideration deeper down that path — so an ID3
path can be at most `n_features` long and the tree is wide and shallow.
CART uses Gini, always splits in two, and may re-test the same feature
repeatedly, which is exactly what lets a binary axis-parallel tree
approximate a diagonal boundary as a staircase. Consequence in practice:
ID3 gives very interpretable "which attribute matters" structure but
cannot express a fine threshold that Gini-CART finds easily, while CART
gives finer boundaries at the cost of deeper trees.

**How CHAID differs from both.** CHAID is not impurity-based at all. It
asks a *statistical* question — "is the association between this predictor
and the target significant?" — and refuses to split when the answer is no.
That makes tree size a consequence of significance rather than of a depth
budget, gives natural protection against splitting on noise, and produces
merged multi-way branches (adjacent value ranges that behave the same are
collapsed into one branch). The Bonferroni adjustment is what stops a
predictor from looking significant merely because many merges were tried.

**Why `ObliqueDecisionTreeModel` was selected as the modern technique.**
All three classical algorithms are *axis-parallel*: every test involves one
feature. A boundary such as `0.9·x0 − 0.7·x1 ≤ 0.2` therefore has to be
approximated by many single-feature cuts, which inflates depth, hurts
stability, and makes the tree look more complicated than the underlying
rule. Oblique (multivariate) trees test a linear combination at each node.
This is the active modern line of decision-tree research — Murthy's OC1,
Wickramarachchi's HHCART (2016), Carreira-Perpiñán & Tavallali's Tree
Alternating Optimization (NeurIPS 2018) and the sparse-oblique work that
followed — and it is the right choice for this project because it is:

* *relevant* — it is a decision tree, not an ensemble, so it does not
  encroach on Group 02 (Random Forest / XGBoost) or any other group,
* *explainable* — L1 regularisation drives most coefficients to exactly
  zero, so a node reads as a short weighted sum of a handful of features,
  and the full hyperplane is exported for the frontend,
* *practical and CPU-friendly* — one small `liblinear` logistic fit per
  node, deterministic under `random_state=42`, no GPU, no extra dependency
  beyond scikit-learn,
* *well matched to this platform* — linear splits need comparable feature
  scales, which the backend's `StandardScaler` step already guarantees.

Because the axis-parallel candidate is always evaluated and wins ties, the
model can never be worse-structured than a greedy CART split at any node.

### Why these defaults

* A bounded depth on all four — no unbounded growth — so training stays
  far inside the five-minute CPU ceiling and the exported tree is small
  enough for the frontend to actually render. The cap is *not* the same
  number for all four, because a level of a multi-way tree costs far more
  nodes than a level of a binary tree: `max_depth=5` for the binary CART
  and oblique trees (≤ 32 leaves), `4` for the multi-way CHAID, `3` for
  the multi-way, never-pruned ID3 (≤ 64 leaves at `n_bins=4`). Measured on
  a 50 000 × 50 four-class matrix this keeps every model at 55–460 nodes
  and a 38–266 KB visualisation payload, instead of the ~1 400-node,
  ~700 KB trees an equal depth of 5 everywhere produces.
* `min_samples_split`/`min_samples_leaf` are `2`/`1` for CART (the textbook
  CART defaults — binary splits at depth 5 cannot blow up anyway), `20`/`1`
  for ID3, `20`/`10` for CHAID and `20`/`5` for the oblique tree, because a
  chi-square test and a logistic fit both need a reasonable number of
  samples to mean anything.
* `n_bins=4` (ID3) — a small alphabet is what ID3 is designed for; larger
  values make each node wider and each branch thinner.
* `max_bins=10` (CHAID) — the standard CHAID practice of starting from
  deciles and letting the merge step decide the real granularity.
* `alpha_merge = alpha_split = 0.05`, `bonferroni=True` — the conventional
  CHAID settings.
* `oblique_C=1.0` — moderate L1 strength: sparse enough to stay readable,
  not so sparse that hyperplanes collapse to a single feature.
* `criterion="gini"` for CART and the oblique tree, entropy for ID3 —
  matching each algorithm's definition.

### How behaviour changes when key choices are altered

| Change | Effect |
|---|---|
| `max_depth` ↑ | More nodes, tighter fit to the training split, slower training, larger visualisation payload; risk of overfitting user data. |
| `min_samples_leaf` ↑ | Fewer, larger leaves; smoother probabilities; fewer splits accepted. |
| `ID3Model(n_bins ↑)` | Wider multi-way splits and finer thresholds, but thinner branches and faster attribute exhaustion (ID3 consumes an attribute per path). |
| `CARTModel(ccp_alpha ↑)` | Progressively prunes the grown tree back; `0.0` keeps it whole. |
| `CARTModel(criterion="entropy")` | Slightly different split choices; behaviour stays CART (binary splits, feature reuse). |
| `CHAIDModel(alpha_split ↓)` | Stricter significance requirement, so a smaller tree — at the limit, a single leaf. |
| `CHAIDModel(alpha_merge ↑)` | More aggressive merging, so fewer and broader branches per split. |
| `CHAIDModel(bonferroni=False)` | Raw p-values are used, so more splits pass; more prone to splitting on noise. |
| `ObliqueDecisionTreeModel(oblique_C ↓)` | Sparser hyperplanes (more zero coefficients), more readable nodes, and more nodes falling back to axis-parallel splits. |
| `ObliqueDecisionTreeModel(criterion="entropy")` | Different threshold choices on the same projections. |

### Determinism

Every model is deterministic: identical `X`, `y` and `random_state=42`
produce an identical tree, identical predictions and an identical
visualisation payload. Ties are broken by the lowest feature index (ID3),
by the stronger chi-square statistic and then the lowest feature index
(CHAID), by the earlier candidate (oblique threshold search), and in favour
of the axis-parallel candidate when an oblique split scores equally.
CHAID compares adjusted p-values exactly rather than against an absolute
tolerance, because on real data those p-values sit many orders of magnitude
below any fixed epsilon and a tolerance would collapse every significant
predictor into one tie.
`random_state=42` is the default on all four classes and is passed to
`liblinear` (oblique) and to scikit-learn's split tie-breaking (CART);
ID3 and CHAID contain no stochastic component at all, and report the seed
purely for interface consistency. `test.py` asserts reproducibility of
predictions, probabilities and tree structure for all four models.

---

## 7. Visualization data

`get_visualization_data()` returns JSON-serialisable dictionaries, lists,
numbers, strings, booleans and `null` only — no figures, no images, no
base64, no file paths. It raises `RuntimeError` if called before `fit()`.

The payload uses the `tree_structure` key required for Group 01, and the
**same schema for all four algorithms** (algorithm-specific fields are
present as `null` where they do not apply), so the frontend needs one
renderer:

```jsonc
{
  "tree_structure": {
    "algorithm": "CART",                  // "ID3" | "CART" | "CHAID" | "Sparse Oblique Decision Tree"
    "split_type": "binary_axis_parallel", // dominant split kind for this algorithm
    "impurity_measure": "gini",           // "gini" | "entropy"
    "root_id": 0,                         // always 0
    "n_nodes": 11,
    "n_leaves": 6,
    "max_depth_reached": 3,
    "n_features": 6,
    "n_oblique_splits": 4,                // ObliqueDecisionTreeModel only
    "feature_names": ["feature_0", "feature_1", "..."],
    "classes": [0, 1],                    // same order as predict_proba columns
    "nodes": [ /* see below */ ],
    "edges": [ /* see below */ ]
  }
}
```

### Node objects

`nodes` is a flat list; `nodes[i]["id"] == i`, and `id` values are
`0 … n_nodes-1`, so the frontend can index directly.

| Field | Type | Meaning |
|---|---|---|
| `id` | int | Node id; the root is `0`. |
| `depth` | int | Distance from the root. |
| `is_leaf` | bool | `true` for leaves. |
| `n_samples` | int | Training samples that reached this node. |
| `impurity` | float | Entropy (ID3, CHAID) or the configured criterion (CART, oblique) at this node. |
| `impurity_measure` | str | `"entropy"` or `"gini"` — which measure `impurity` is. |
| `class_distribution` | dict | `{class_label_as_string: count}` for every class. |
| `class_probabilities` | list[float] | Same order as `classes`; sums to 1. This is what a sample landing here gets from `predict_proba`. |
| `predicted_class` | scalar | The label `predict` returns for this node. |
| `split` | dict \| null | `null` for leaves, otherwise the split description below. |
| `children` | list[int] | Child node ids, ordered to match `split`'s branches (empty for leaves). |

### Split objects

Every split object carries **all** of these keys; the ones that do not
apply to the algorithm are `null`.

| Field | Type | Present for | Meaning |
|---|---|---|---|
| `type` | str | all | `"multiway_binned"` (ID3), `"binary_axis_parallel"` (CART, and oblique fallback nodes), `"multiway_chi_square"` (CHAID), `"binary_oblique"` (oblique). |
| `feature` | int \| null | all except oblique hyperplane nodes | Column index tested. |
| `feature_name` | str \| null | as above | `"feature_{index}"`. |
| `gain` | float | all | Weighted impurity decrease produced by this split (information gain for ID3/CHAID, Gini/entropy decrease for CART/oblique). |
| `threshold` | float \| null | CART, oblique | Right-hand side of `... <= threshold`. |
| `bin_edges` | list[float] \| null | ID3, CHAID | Interior bin edges defining the ordinal categories. |
| `coefficients` | list[float] \| null | oblique | Hyperplane weights `w`, length `n_features` (mostly zeros). |
| `intercept` | float \| null | oblique | Hyperplane offset `b`. |
| `chi_square` | float \| null | CHAID | Pearson chi-square statistic of the merged table. |
| `p_value` | float \| null | CHAID | Raw p-value. |
| `p_value_adjusted` | float \| null | CHAID | Bonferroni-adjusted p-value (the value compared against `alpha_split`). |
| `degrees_of_freedom` | int \| null | CHAID | Degrees of freedom of the chi-square test. |
| `condition` | str | all | Ready-to-display rule text for the node, e.g. `"feature_2 <= 0.4131"`, `"multi-way split on feature_0 (4 bins)"`, `"chi-square split on feature_1 (3 merged groups, adjusted p=0.0002)"`, `"(-9.3611*feature_0 +6.8922*feature_1) -0.2290 <= 0.0251"`. |

### Edge objects

`edges` lists every parent → child link, and is exactly consistent with
the `children` arrays.

| Field | Type | Meaning |
|---|---|---|
| `source` | int | Parent node id. |
| `target` | int | Child node id. |
| `branch_index` | int | Position of this branch in the parent's `children`. |
| `label` | str | The condition satisfied by samples following this edge — e.g. `"feature_2 <= 0.4131"` / `"feature_2 > 0.4131"` for binary splits, `"0.1234 < feature_0 <= 0.8765"` for a binned or merged branch, and the signed linear expression compared against the threshold for oblique splits. |

### Rendering notes

* A node-link diagram can be drawn directly from `nodes` + `edges`; use
  `n_samples` for node size, `class_probabilities` for a stacked bar or pie
  inside the node, `predicted_class` for the leaf label and
  `split["condition"]` or `edge["label"]` for the rule text.
* Branch counts vary: CART and the oblique tree always have 2 children,
  ID3 and CHAID have 2 or more, so the renderer should not assume binary.
* `feature_importance` in `get_metadata()` uses the same `feature_{i}`
  names and is normalised to sum to 1 (all zeros if the tree is a single
  leaf), so it can drive a bar chart alongside the tree.

---

## 8. Integration notes

Nothing outside this folder was touched: `base_model.py`, the registry,
`backend/tests/`, the frontend and the CI workflow are unmodified.

**Import.** All four classes are re-exported from the package, so import
from the package and not from the file layout:

```python
from models.group_01_decision_trees import (
    CARTModel,
    CHAIDModel,
    ID3Model,
    ObliqueDecisionTreeModel,
)
```

**Registry entries** (for the integration team to add — we have not edited
the registry):

| Dropdown label | Class | Import path |
|---|---|---|
| `ID3 Decision Tree` | `ID3Model` | `models.group_01_decision_trees.id3` |
| `CART Decision Tree` | `CARTModel` | `models.group_01_decision_trees.cart` |
| `CHAID Decision Tree` | `CHAIDModel` | `models.group_01_decision_trees.chaid` |
| `Sparse Oblique Decision Tree` | `ObliqueDecisionTreeModel` | `models.group_01_decision_trees.oblique_tree` |

The dropdown labels above are exactly the `model_name` values returned by
`get_metadata()`.

**Platform flow.** Each class supports the full request cycle with no
special-casing:

```python
model = REGISTRY[selected_name]()        # no required constructor arguments
model.fit(X_train, y_train)              # returns self
predictions = model.predict(X_test)      # 1D, labels
probabilities = model.predict_proba(X_test)  # (n, n_classes), rows sum to 1
metadata = model.get_metadata()          # exactly the six required keys
visualisation = model.get_visualization_data()   # {"tree_structure": {...}}
```

**Error handling.** Unsuitable input raises a specific exception with a
readable message for the backend to surface — never `None`, never an empty
array, never a bare assertion:

* `ValueError` — `y is None`, `X` not 2D, `y` not 1D, row-count mismatch,
  empty `X`, non-finite values, wrong feature count at predict time, or an
  invalid hyperparameter value at construction time.
* `RuntimeError` — `predict`, `predict_proba` or `get_visualization_data`
  called before `fit`.

**Optional constructor arguments.** All hyperparameters have defaults, and
every class accepts `**kwargs`, so the registry can pass user-selected
hyperparameters through without the constructor signature having to match.
Unknown extras are recorded in `hyperparams` and therefore appear in
`get_metadata()["hyperparameters"]`.

**Performance.** All four are pure CPU, single-threaded except where
scikit-learn parallelises internally. Measured on a laptop CPU with
default hyperparameters (80/20 split, fit time on the training part):

| Dataset | ID3 | CART | CHAID | Oblique |
|---|---|---|---|---|
| 1 000 × 10, 2 classes | 0.01 s | 0.01 s | 0.5 s | 0.02 s |
| 10 000 × 30, 3 classes | 0.08 s | 0.29 s | 5.1 s | 1.4 s |
| 50 000 × 50, 4 classes | 0.5 s | 2.6 s | 11.1 s | 24.8 s |

All far inside the five-minute ceiling. CHAID is the most expensive on
wide data (it runs a merge-and-test procedure per feature per node) and
the oblique tree on large data (one small logistic fit per node); if a
user uploads something much larger, lowering `max_depth` or raising
`min_samples_split` reduces both roughly linearly. `predict`,
`predict_proba` and `get_visualization_data` are sub-second in every case
above; the visualisation payload ranges from 10 KB to 266 KB of JSON.
