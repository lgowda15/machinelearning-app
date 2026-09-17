import { describe, expect, it } from "vitest";
import { renderReportPdf } from "./renderReportPdf";
import type { ReportModel } from "./types";

function minimalModel(overrides: Partial<ReportModel> = {}): ReportModel {
  return {
    summary: {
      datasetName: "iris.csv",
      nRows: 100,
      nCols: 3,
      target: "target",
      dataType: "classification",
      testSize: 0.2,
      modelDisplayNames: ["Logistic Regression"],
      generatedAt: new Date("2026-01-01T00:00:00Z").toISOString(),
    },
    preprocessing: {},
    dataset: {
      imbalance: undefined,
      columns: [],
      truncatedColumnsCount: 0,
      distributionCharts: [],
    },
    models: [
      {
        key: "logistic_regression",
        displayName: "Logistic Regression",
        modelType: "classifier",
        hyperparameters: { C: 1.0 },
        metrics: { accuracy: 0.9 },
        charts: [],
      },
    ],
    comparison: null,
    prediction: null,
    ...overrides,
  };
}

describe("renderReportPdf", () => {
  it("resolves to a PDF blob for a minimal report with no charts", async () => {
    const blob = await renderReportPdf(minimalModel());
    expect(blob).toBeInstanceOf(Blob);
    expect(blob.type).toBe("application/pdf");
    expect(blob.size).toBeGreaterThan(0);
  });

  it("resolves to a PDF blob when charts, comparison, and prediction are all present", async () => {
    const model = minimalModel({
      dataset: {
        imbalance: { isImbalanced: true, message: "Minority class is under 15%." },
        columns: [
          {
            name: "a",
            dtype: "numeric",
            is_target: false,
            missing_count: 0,
            missing_pct: 0,
            unique_count: 5,
            distribution: { kind: "numeric", bin_edges: [0, 1], counts: [1] },
          },
        ],
        truncatedColumnsCount: 0,
        distributionCharts: [
          {
            kind: "distribution",
            title: "a",
            column: {
              name: "a",
              dtype: "numeric",
              is_target: false,
              missing_count: 0,
              missing_pct: 0,
              unique_count: 5,
              distribution: { kind: "numeric", bin_edges: [0, 1], counts: [1] },
            },
            targetAccentVar: "var(--color-signal)",
          },
        ],
      },
      models: [
        {
          key: "logistic_regression",
          displayName: "Logistic Regression",
          modelType: "classifier",
          hyperparameters: { C: 1.0 },
          metrics: { accuracy: 0.9, confusion_matrix: [[8, 1], [1, 10]], labels: ["0", "1"] },
          charts: [
            {
              kind: "confusion-matrix",
              title: "Confusion matrix",
              confusionMatrix: [[8, 1], [1, 10]],
              labels: ["0", "1"],
            },
            {
              kind: "feature-importance",
              title: "Feature importance",
              featureImportance: { a: 0.6, b: 0.4 },
            },
            {
              kind: "tree-summary",
              title: "Tree structure",
              depth: 2,
              nodeCount: 3,
              leafCount: 2,
              rules: ["feature_0 <= 0.5 → class 1"],
            },
          ],
        },
      ],
      comparison: {
        modelKeys: ["logistic_regression", "other_classifier"],
        metricsTable: { headers: ["Metric", "Logistic Regression", "Other Classifier"], rows: [["accuracy", "0.9000", "0.8000"]] },
      },
      prediction: {
        modelKey: "logistic_regression",
        inputSummary: "CSV upload: new.csv",
        totalRows: 2,
        outputSample: { headers: ["Row", "Prediction"], rows: [[0, "0"], [1, "1"]] },
        predictedValueDistribution: { headers: ["Predicted value", "Count"], rows: [["0", 1], ["1", 1]] },
      },
    });

    const blob = await renderReportPdf(model);
    expect(blob).toBeInstanceOf(Blob);
    expect(blob.type).toBe("application/pdf");
    expect(blob.size).toBeGreaterThan(0);
  });
});
