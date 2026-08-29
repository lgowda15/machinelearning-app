import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ConfusionMatrixChart } from "./ConfusionMatrixChart";

describe("ConfusionMatrixChart", () => {
  it("renders one row/column per label and the raw counts", () => {
    render(
      <ConfusionMatrixChart
        confusionMatrix={[
          [2, 0, 0],
          [0, 2, 0],
          [0, 0, 2],
        ]}
        labels={["0", "1", "2"]}
      />,
    );
    expect(screen.getAllByRole("cell", { name: "2" })).toHaveLength(3); // the three diagonal cells
    expect(screen.getByText("6 of 6 test samples correctly classified.")).toBeInTheDocument();
  });

  it("counts off-diagonal cells as incorrect in the caption", () => {
    render(
      <ConfusionMatrixChart
        confusionMatrix={[
          [1, 1],
          [0, 2],
        ]}
        labels={["a", "b"]}
      />,
    );
    expect(screen.getByText("3 of 4 test samples correctly classified.")).toBeInTheDocument();
  });
});
