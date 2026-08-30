/**
 * Shapes for the three `get_visualization_data()` custom payloads that
 * don't yet have a frontend component (CLAUDE.md "Known gap"; contract
 * described loosely in DATA_FLOW_GUIDE.md §5.3 as "whatever JSON-safe
 * structure they return"). `TrainedModelResponse.visualization_data` is
 * `dict[str, Any] | None` on the backend (app/schemas/training.py) --
 * deliberately not typed further there, since format isn't uniform across
 * models by design. These interfaces are NOT derived from the OpenAPI
 * schema (there is nothing there to derive from); they were captured by
 * fitting each group's actual model and inspecting the real
 * `get_visualization_data()` output:
 *
 * - group_01 (CART/CHAID/ID3/oblique tree, all four): `tree_structure`
 * - group_02 (Random Forest, XGBoost): `shap_values`
 * - group_09 (Hierarchical Clustering): `linkage_matrix`
 */

/** group_01_decision_trees -- `{ tree_structure: TreeStructurePayload }`.
 *
 * One schema shared by all four tree algorithms; `split.type` is the
 * discriminator ("binary_axis_parallel" for CART/ID3/oblique's axis
 * splits, "binary_oblique" for oblique's hyperplane splits -- only these
 * carry non-null `coefficients` -- "multiway_chi_square" for CHAID,
 * "multiway_binned" for ID3 -- only these carry non-null `chi_square` /
 * `p_value` and, for CHAID, `p_value_adjusted` / `degrees_of_freedom`, and
 * may have more than two `children`). `condition` and each edge's `label`
 * are always a ready-to-display string regardless of split type --
 * confirmed by group_01's own test.py, which asserts `condition` is a str
 * on every node and every edge has exactly the four keys below -- so the
 * tree view never needs to branch on split type to render the diagram
 * itself, only to decide which extra fields (chi-square, coefficients) are
 * worth a caption.
 */
export interface TreeSplit {
  type: string;
  feature: number;
  feature_name: string;
  gain: number | null;
  threshold: number | null;
  bin_edges: number[] | null;
  coefficients: number[] | null;
  intercept: number | null;
  chi_square: number | null;
  p_value: number | null;
  p_value_adjusted: number | null;
  degrees_of_freedom: number | null;
  condition: string;
}

export interface TreeNode {
  id: number;
  depth: number;
  is_leaf: boolean;
  n_samples: number;
  impurity: number;
  impurity_measure: string;
  class_distribution: Record<string, number>;
  class_probabilities: number[];
  predicted_class: string | number;
  split: TreeSplit | null;
  /** Child node ids, in branch order. Empty for a leaf. */
  children: number[];
}

export interface TreeEdge {
  source: number;
  target: number;
  branch_index: number;
  label: string;
}

export interface TreeStructurePayload {
  algorithm: string;
  split_type: string;
  impurity_measure: string;
  root_id: number;
  n_nodes: number;
  n_leaves: number;
  max_depth_reached: number;
  n_features: number;
  feature_names: string[];
  classes: (string | number)[];
  nodes: TreeNode[];
  edges: TreeEdge[];
}

/** group_02_random_forest_xgboost -- `{ shap_values: ShapValues }`.
 *
 * `shap.TreeExplainer(...).shap_values(X)` on the installed shap version
 * (see backend/.venv) returns a single ndarray of shape
 * `(n_samples, n_features, n_classes)` for a classifier -- confirmed by
 * fitting both RandomForestModel and XGBoostModel on binary (n_classes=2)
 * and multiclass (n_classes=3) data and inspecting the real output; both
 * shapes came back 3D, never the older "list of one 2D array per class"
 * form. `_shap_to_nested_lists` (random_forest.py / xgboost_model.py)
 * still branches on `isinstance(shap_values, list)` for older shap
 * versions, so a plain `(n_samples, n_features)` 2D array is kept here as
 * a fallback shape rather than assumed away.
 */
export type ShapValues = number[][][] | number[][];

/** group_09_dbscan_hierarchical (HierarchicalClusteringModel only --
 * DBSCANModel has no `get_visualization_data`) -- `{ linkage_matrix }`.
 *
 * Exactly `scipy.cluster.hierarchy.linkage`'s own output contract, per the
 * model's own docstring: `(n_samples - 1) x 4`, each row
 * `[idx1, idx2, distance, sample_count]`. `idx1`/`idx2` index into the
 * combined leaves-then-merges numbering SciPy uses: `0..n-1` are the
 * original samples, `n..2n-2` refer back into this same matrix
 * (`n + row_index`).
 */
export type LinkageMatrix = number[][];
