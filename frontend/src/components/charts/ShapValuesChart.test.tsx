import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ShapValuesChart } from "./ShapValuesChart";
import type { ShapValues } from "../../types/visualizationData";

// (n_samples=2, n_features=2, n_classes=2). feature_0's mean |SHAP| (1.5)
// dominates feature_1's (0.15) for both classes.
const SHAP_VALUES: ShapValues = [
  [
    [1, -1],
    [0.2, -0.2],
  ],
  [
    [2, -2],
    [0.1, -0.1],
  ],
];

describe("ShapValuesChart", () => {
  it("names the highest mean |SHAP| feature and reports the sample/class counts", () => {
    render(
      <ShapValuesChart shapValues={SHAP_VALUES} featureNames={["feature_0", "feature_1"]} classLabels={["0", "1"]} />,
    );
    expect(
      screen.getByText("Mean |SHAP value| across 2 test samples, 2 classes. Top feature: feature_0."),
    ).toBeInTheDocument();
  });

  it("falls back to a single series for a 2D (n_samples, n_features) array", () => {
    const flat: ShapValues = [
      [1, 0.2],
      [2, 0.1],
    ];
    render(<ShapValuesChart shapValues={flat} featureNames={["feature_0", "feature_1"]} classLabels={[]} />);
    expect(screen.getByText("Mean |SHAP value| across 2 test samples. Top feature: feature_0.")).toBeInTheDocument();
  });
});
