import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { TreeStructureChart } from "./TreeStructureChart";
import type { TreeStructurePayload } from "../../types/visualizationData";

const TREE: TreeStructurePayload = {
  algorithm: "CART",
  split_type: "binary_axis_parallel",
  impurity_measure: "gini",
  root_id: 0,
  n_nodes: 3,
  n_leaves: 2,
  max_depth_reached: 1,
  n_features: 2,
  feature_names: ["feature_0", "feature_1"],
  classes: [0, 1],
  nodes: [
    {
      id: 0,
      depth: 0,
      is_leaf: false,
      n_samples: 10,
      impurity: 0.5,
      impurity_measure: "gini",
      class_distribution: { "0": 5, "1": 5 },
      class_probabilities: [0.5, 0.5],
      predicted_class: 0,
      split: {
        type: "binary_axis_parallel",
        feature: 0,
        feature_name: "feature_0",
        gain: 0.5,
        threshold: 0.1,
        bin_edges: null,
        coefficients: null,
        intercept: null,
        chi_square: null,
        p_value: null,
        p_value_adjusted: null,
        degrees_of_freedom: null,
        condition: "feature_0 <= 0.1000",
      },
      children: [1, 2],
    },
    {
      id: 1,
      depth: 1,
      is_leaf: true,
      n_samples: 5,
      impurity: 0,
      impurity_measure: "gini",
      class_distribution: { "0": 5, "1": 0 },
      class_probabilities: [1, 0],
      predicted_class: 0,
      split: null,
      children: [],
    },
    {
      id: 2,
      depth: 1,
      is_leaf: true,
      n_samples: 5,
      impurity: 0,
      impurity_measure: "gini",
      class_distribution: { "0": 0, "1": 5 },
      class_probabilities: [0, 1],
      predicted_class: 1,
      split: null,
      children: [],
    },
  ],
  edges: [
    { source: 0, target: 1, branch_index: 0, label: "feature_0 <= 0.1000" },
    { source: 0, target: 2, branch_index: 1, label: "feature_0 > 0.1000" },
  ],
};

describe("TreeStructureChart", () => {
  it("renders the root split condition and both leaves' predicted classes", () => {
    render(<TreeStructureChart data={TREE} />);
    // Appears twice: the root node's own condition, and the left branch's
    // edge label -- both algorithms use the identical string for this tree.
    expect(screen.getAllByText("feature_0 <= 0.1000")).toHaveLength(2);
    expect(screen.getByText("class 0")).toBeInTheDocument();
    expect(screen.getByText("class 1")).toBeInTheDocument();
  });

  it("labels the right branch with its own edge condition", () => {
    render(<TreeStructureChart data={TREE} />);
    expect(screen.getByText("feature_0 > 0.1000")).toBeInTheDocument();
  });

  it("summarises node/leaf/depth counts in the caption", () => {
    render(<TreeStructureChart data={TREE} />);
    expect(screen.getByText("CART: 3 nodes, 2 leaves, depth 1, gini impurity.")).toBeInTheDocument();
  });
});
