import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { DendrogramChart } from "./DendrogramChart";

// scipy.cluster.hierarchy.linkage contract: [idx1, idx2, distance, count].
// 3 leaves (0, 1, 2): merge 0+1 at 0.5 -> node 3, then node 3 + leaf 2 at 1.0.
const LINKAGE_MATRIX = [
  [0, 1, 0.5, 2],
  [3, 2, 1.0, 3],
];

describe("DendrogramChart", () => {
  it("labels the SVG with the leaf count and captions sample/merge counts", () => {
    render(<DendrogramChart linkageMatrix={LINKAGE_MATRIX} />);
    expect(screen.getByRole("img", { name: "Dendrogram of 3 samples" })).toBeInTheDocument();
    expect(screen.getByText("3 samples, 2 merges, max distance 1.000.")).toBeInTheDocument();
  });

  it("omits the leaf-labels caveat when leaf count is small", () => {
    render(<DendrogramChart linkageMatrix={LINKAGE_MATRIX} />);
    expect(screen.queryByText(/leaf labels omitted/)).not.toBeInTheDocument();
  });
});
