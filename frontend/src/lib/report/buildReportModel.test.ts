import { describe, expect, it } from "vitest";
import { buildReportModel } from "./buildReportModel";
import type { CycleState } from "./types";
import type { DataProfileResponse } from "../../hooks/useDataset";
import type { components } from "../../types/api";

type TrainResponse = components["schemas"]["TrainResponse"];
type ComparisonResponse = components["schemas"]["ComparisonResponse"];
type PredictionResponse = components["schemas"]["PredictionResponse"];
type ColumnSummary = components["schemas"]["ColumnSummary"];

function numericColumn(name: string, isTarget = false): ColumnSummary {
  return {
    name,
    dtype: "numeric",
    is_target: isTarget,
    missing_count: 0,
    missing_pct: 0,
    unique_count: 10,
    distribution: { kind: "numeric", bin_edges: [0, 1], counts: [1] },
  };
}

function profile(overrides: Partial<DataProfileResponse> = {}): DataProfileResponse {
  return {
    data_id: "d1",
    source: "iris.csv",
    n_rows: 100,
    n_columns: 3,
    data_type: "classification",
    target_column: "target",
    columns: [numericColumn("a"), numericColumn("b"), numericColumn("target", true)],
    ...overrides,
  };
}

function trainResponse(overrides: Partial<TrainResponse> = {}): TrainResponse {
  return {
    training_id: "t1",
    data_id: "d1",
    test_size: 0.2,
    results: [
      {
        model_key: "logistic_regression",
        model_name: "Logistic Regression",
        model_type: "classifier",
        metrics: { accuracy: 0.9, confusion_matrix: [[8, 1], [1, 10]], labels: ["0", "1"] },
        hyperparameters: { C: 1.0 },
        training_time_seconds: 0.01,
        n_features: 2,
        feature_importance: null,
        visualization_data: null,
        plot_data: null,
      },
      {
        model_key: "other_classifier",
        model_name: "Other Classifier",
        model_type: "classifier",
        metrics: { accuracy: 0.8, confusion_matrix: [[7, 2], [2, 9]], labels: ["0", "1"] },
        hyperparameters: {},
        training_time_seconds: 0.01,
        n_features: 2,
        feature_importance: null,
        visualization_data: null,
        plot_data: null,
      },
    ],
    ...overrides,
  };
}

function baseCycle(overrides: Partial<CycleState> = {}): CycleState {
  return {
    profile: profile(),
    testSize: 0.2,
    trainingResults: trainResponse(),
    comparisonResult: null,
    prediction: null,
    ...overrides,
  };
}

function comparisonResponse(): ComparisonResponse {
  return {
    common_metrics: ["accuracy"],
    models: [
      {
        training_id: "t1",
        model_key: "logistic_regression",
        model_name: "Logistic Regression",
        model_type: "classifier",
        metrics: { accuracy: 0.9 },
      },
      {
        training_id: "t1",
        model_key: "other_classifier",
        model_name: "Other Classifier",
        model_type: "classifier",
        metrics: { accuracy: 0.8 },
      },
    ],
  };
}

function predictionResponse(overrides: Partial<PredictionResponse> = {}): PredictionResponse {
  return {
    training_id: "t1",
    model_key: "logistic_regression",
    model_type: "classifier",
    n_samples: 2,
    predictions: [0, 1],
    probabilities: [
      [0.9, 0.1],
      [0.2, 0.8],
    ],
    ...overrides,
  };
}

describe("buildReportModel", () => {
  it("builds a results-only report, omitting comparison and prediction entirely", () => {
    const model = buildReportModel(baseCycle());

    expect(model.summary.datasetName).toBe("iris.csv");
    expect(model.summary.modelDisplayNames).toEqual(["Logistic Regression", "Other Classifier"]);
    expect(model.models).toHaveLength(2);
    expect(model.models[0].charts.some((c) => c.kind === "confusion-matrix")).toBe(true);
    expect(model.comparison).toBeNull();
    expect(model.prediction).toBeNull();
  });

  it("includes a comparison section re-derived from trainingResults when a comparison was run", () => {
    const model = buildReportModel(baseCycle({ comparisonResult: comparisonResponse() }));

    expect(model.comparison).not.toBeNull();
    expect(model.comparison!.modelKeys).toEqual(["logistic_regression", "other_classifier"]);
    expect(model.comparison!.metricsTable.headers).toEqual(["Metric", "Logistic Regression", "Other Classifier"]);
    // Values come from trainingResults, not the (deliberately different)
    // comparisonResult.models[].metrics passed in above.
    expect(model.comparison!.metricsTable.rows).toEqual([["accuracy", "0.9000", "0.8000"]]);
    expect(model.prediction).toBeNull();
  });

  it("includes a prediction section when a prediction was run", () => {
    const model = buildReportModel(
      baseCycle({
        prediction: { inputSummary: "CSV upload: new.csv", result: predictionResponse() },
      }),
    );

    expect(model.prediction).not.toBeNull();
    expect(model.prediction!.modelKey).toBe("logistic_regression");
    expect(model.prediction!.inputSummary).toBe("CSV upload: new.csv");
    expect(model.prediction!.totalRows).toBe(2);
    expect(model.comparison).toBeNull();
  });

  it("includes both comparison and prediction for a full cycle", () => {
    const model = buildReportModel(
      baseCycle({
        comparisonResult: comparisonResponse(),
        prediction: { inputSummary: "Manual entry", result: predictionResponse() },
      }),
    );

    expect(model.comparison).not.toBeNull();
    expect(model.prediction).not.toBeNull();
  });

  it("carries imbalance info through when present", () => {
    const model = buildReportModel(
      baseCycle({
        profile: profile({
          class_imbalance: { is_imbalanced: true, class_counts: { "0": 90, "1": 10 }, message: "Minority class is under 15%." },
        }),
      }),
    );

    expect(model.dataset.imbalance).toEqual({ isImbalanced: true, message: "Minority class is under 15%." });
  });

  it("omits imbalance info when absent", () => {
    const model = buildReportModel(baseCycle());
    expect(model.dataset.imbalance).toBeUndefined();
  });

  it("caps the column list at 20 and reports how many were omitted", () => {
    const manyColumns = Array.from({ length: 25 }, (_, i) => numericColumn(`col_${i}`));
    const model = buildReportModel(
      baseCycle({ profile: profile({ columns: manyColumns, target_column: null }) }),
    );

    expect(model.dataset.columns).toHaveLength(20);
    expect(model.dataset.truncatedColumnsCount).toBe(5);
    expect(model.dataset.distributionCharts).toHaveLength(20);
  });

  it("does not truncate when there are 20 columns or fewer", () => {
    const model = buildReportModel(baseCycle());
    expect(model.dataset.truncatedColumnsCount).toBe(0);
  });

  it("caps the prediction output sample at 25 rows while keeping the full row count and value distribution", () => {
    const predictions = Array.from({ length: 40 }, (_, i) => i % 2);
    const model = buildReportModel(
      baseCycle({
        prediction: {
          inputSummary: "CSV upload: big.csv",
          result: predictionResponse({ n_samples: 40, predictions, probabilities: null }),
        },
      }),
    );

    expect(model.prediction!.totalRows).toBe(40);
    expect(model.prediction!.outputSample.rows).toHaveLength(25);
    expect(model.prediction!.predictedValueDistribution.rows).toEqual(
      expect.arrayContaining([
        ["0", 20],
        ["1", 20],
      ]),
    );
  });

  it("never resurfaces a prediction/comparison made against a training run that has since been invalidated", () => {
    // Simulates App's resetCycle firing (a re-upload, a re-profile, or a
    // model-selection change) between an earlier prediction/comparison and
    // the next report build: the reset clears comparisonResult/prediction
    // before a new training run lands, so the cycle handed to
    // buildReportModel here has a *new* training run with neither -- there
    // is no path back to the stale run's report sections.
    const cycleAfterReset = baseCycle({
      trainingResults: trainResponse({ training_id: "t2" }),
      comparisonResult: null,
      prediction: null,
    });

    const model = buildReportModel(cycleAfterReset);

    expect(model.comparison).toBeNull();
    expect(model.prediction).toBeNull();
  });

  it("throws if a comparison references a model outside the current training run", () => {
    const staleComparison = comparisonResponse();
    expect(() =>
      buildReportModel(
        baseCycle({
          trainingResults: trainResponse({
            results: [trainResponse().results[0]],
          }),
          comparisonResult: staleComparison,
        }),
      ),
    ).toThrow(/other_classifier/);
  });

  it("throws when there is no profile or no training results", () => {
    expect(() => buildReportModel(baseCycle({ profile: null }))).toThrow(/profile/);
    expect(() => buildReportModel(baseCycle({ trainingResults: null }))).toThrow(/trained model/);
  });
});
