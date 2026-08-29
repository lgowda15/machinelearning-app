import { describe, expect, it } from "vitest";
import { bucketsForColumn, topBucketCaption } from "./distribution";
import type { components } from "../types/api";

type ColumnSummary = components["schemas"]["ColumnSummary"];

function numericColumn(): ColumnSummary {
  return {
    name: "age",
    dtype: "numeric",
    is_target: false,
    missing_count: 0,
    missing_pct: 0,
    unique_count: 5,
    distribution: {
      kind: "numeric",
      bin_edges: [0, 10, 20, 30],
      counts: [2, 5, 1],
    },
  };
}

function categoricalColumn(otherCount = 0): ColumnSummary {
  return {
    name: "species",
    dtype: "categorical",
    is_target: true,
    missing_count: 0,
    missing_pct: 0,
    unique_count: 2,
    distribution: {
      kind: "categorical",
      categories: ["setosa", "versicolor"],
      counts: [30, 20],
      other_count: otherCount,
    },
  };
}

describe("bucketsForColumn", () => {
  it("builds one bucket per numeric bin, labelled by its edges", () => {
    const buckets = bucketsForColumn(numericColumn());
    expect(buckets).toEqual([
      { label: "0.0–10.0", count: 2 },
      { label: "10.0–20.0", count: 5 },
      { label: "20.0–30.0", count: 1 },
    ]);
  });

  it("builds one bucket per category, in the order the backend returned", () => {
    const buckets = bucketsForColumn(categoricalColumn());
    expect(buckets).toEqual([
      { label: "setosa", count: 30 },
      { label: "versicolor", count: 20 },
    ]);
  });

  it("appends an Other bucket only when other_count is positive", () => {
    const buckets = bucketsForColumn(categoricalColumn(7));
    expect(buckets).toHaveLength(3);
    expect(buckets[2]).toEqual({ label: "Other", count: 7 });
  });
});

describe("topBucketCaption", () => {
  it("names the largest bucket", () => {
    expect(topBucketCaption(bucketsForColumn(numericColumn()))).toBe(
      "Most common: 10.0–20.0 (n=5)",
    );
  });

  it("returns null with no buckets", () => {
    expect(topBucketCaption([])).toBeNull();
  });
});
