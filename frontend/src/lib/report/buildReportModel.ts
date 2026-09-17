import { formatMetricValue, formatPredictionValue } from "../format";
import { targetColorVarForDataType } from "../modelType";
import { buildTreeSummaryRules } from "./treeSummaryTable";
import type { components } from "../../types/api";
import type { LinkageMatrix, ShapValues, TreeStructurePayload } from "../../types/visualizationData";
import type {
  ChartSpec,
  CycleState,
  ReportModel,
  ReportModelModel,
  TableSpec,
} from "./types";

type TrainedModelResponse = components["schemas"]["TrainedModelResponse"];

const MAX_COLUMNS = 20;
const MAX_PREDICTION_ROWS_SHOWN = 25;

export function buildReportModel(cycle: CycleState): ReportModel {
  const { profile, testSize, trainingResults, comparisonResult, prediction } = cycle;

  if (!profile) {
    throw new Error("buildReportModel requires a dataset profile");
  }
  if (!trainingResults || trainingResults.results.length === 0) {
    throw new Error("buildReportModel requires at least one trained model");
  }

  return {
    summary: {
      datasetName: profile.source,
      nRows: profile.n_rows,
      nCols: profile.n_columns,
      target: profile.target_column ?? null,
      dataType: profile.data_type,
      testSize,
      modelDisplayNames: trainingResults.results.map((r) => r.model_name),
      generatedAt: new Date().toISOString(),
    },
    preprocessing: {},
    dataset: buildDatasetSection(profile),
    models: trainingResults.results.map(buildModelSection),
    comparison: comparisonResult ? buildComparisonSection(comparisonResult, trainingResults.results) : null,
    prediction: prediction ? buildPredictionSection(prediction, trainingResults.results) : null,
  };
}

/* -------------------------------------------------------------------------- */
/* Dataset section                                                            */
/* -------------------------------------------------------------------------- */

function buildDatasetSection(profile: NonNullable<CycleState["profile"]>): ReportModel["dataset"] {
  // Target column first, same ordering EdaScreen uses.
  const orderedColumns = [
    ...profile.columns.filter((c) => c.is_target),
    ...profile.columns.filter((c) => !c.is_target),
  ];

  const columns = orderedColumns.slice(0, MAX_COLUMNS);
  const truncatedColumnsCount = Math.max(0, orderedColumns.length - MAX_COLUMNS);
  const targetAccentVar = targetColorVarForDataType(profile.data_type);

  const distributionCharts: ChartSpec[] = columns.map((column) => ({
    kind: "distribution",
    title: column.is_target ? `${column.name} (target)` : column.name,
    column,
    targetAccentVar,
  }));

  return {
    imbalance: profile.class_imbalance
      ? {
          isImbalanced: profile.class_imbalance.is_imbalanced,
          message: profile.class_imbalance.message ?? undefined,
        }
      : undefined,
    columns,
    truncatedColumnsCount,
    distributionCharts,
  };
}

/* -------------------------------------------------------------------------- */
/* Per-model section                                                          */
/* -------------------------------------------------------------------------- */

function buildModelSection(result: TrainedModelResponse): ReportModelModel {
  const metrics = result.metrics as Record<string, unknown>;
  const charts: ChartSpec[] = [];

  switch (result.model_type) {
    case "classifier": {
      const confusionMatrix = metrics.confusion_matrix;
      if (Array.isArray(confusionMatrix)) {
        charts.push({
          kind: "confusion-matrix",
          title: "Confusion matrix",
          confusionMatrix: confusionMatrix as number[][],
          labels: (metrics.labels as string[] | undefined) ?? [],
        });
      }
      break;
    }

    case "clusterer": {
      const plotData = result.plot_data as { points: number[][]; labels: number[] } | null;
      if (plotData) {
        charts.push({
          kind: "cluster-scatter",
          title: "Cluster assignments",
          points: plotData.points,
          labels: plotData.labels,
        });
      }
      break;
    }

    case "regressor": {
      const plotData = result.plot_data as { y_true: number[]; y_pred: number[] } | null;
      if (plotData) {
        charts.push({
          kind: "predicted-vs-actual",
          title: "Predicted vs. actual",
          yTrue: plotData.y_true,
          yPred: plotData.y_pred,
        });
      }
      break;
    }

    case "dimensionality_reducer": {
      const explainedVarianceRatio = metrics.explained_variance_ratio;
      if (Array.isArray(explainedVarianceRatio)) {
        charts.push({
          kind: "variance-plot",
          title: "Explained variance",
          explainedVarianceRatio: explainedVarianceRatio as (number | null)[],
        });
      }
      break;
    }
  }

  if (result.feature_importance) {
    charts.push({
      kind: "feature-importance",
      title: "Feature importance",
      featureImportance: result.feature_importance,
    });
  }

  if (result.visualization_data) {
    const data = result.visualization_data;

    if ("tree_structure" in data) {
      const tree = data.tree_structure as TreeStructurePayload;
      charts.push({
        kind: "tree-summary",
        title: "Tree structure",
        depth: tree.max_depth_reached,
        nodeCount: tree.n_nodes,
        leafCount: tree.n_leaves,
        rules: buildTreeSummaryRules(tree),
      });
    } else if ("shap_values" in data) {
      const featureNames = Array.from({ length: result.n_features }, (_, i) => `feature_${i}`);
      const classLabels = (metrics.labels as string[] | undefined) ?? [];

      charts.push({
        kind: "shap-values",
        title: "SHAP values",
        shapValues: data.shap_values as ShapValues,
        featureNames,
        classLabels,
      });
    } else if ("linkage_matrix" in data) {
      charts.push({
        kind: "dendrogram",
        title: "Dendrogram",
        linkageMatrix: data.linkage_matrix as LinkageMatrix,
      });
    }
    // Any other custom payload (CLAUDE.md's "known gap") has no chart yet --
    // its raw metrics/hyperparameters still appear in the model's tables.
  }

  return {
    key: result.model_key,
    displayName: result.model_name,
    modelType: result.model_type,
    hyperparameters: result.hyperparameters,
    metrics: result.metrics,
    charts,
  };
}

/* -------------------------------------------------------------------------- */
/* Comparison section                                                         */
/* -------------------------------------------------------------------------- */

function buildComparisonSection(
  comparisonResult: NonNullable<CycleState["comparisonResult"]>,
  trainedResults: TrainedModelResponse[],
): ReportModel["comparison"] {
  const byKey = new Map(trainedResults.map((r) => [r.model_key, r]));
  const modelKeys = comparisonResult.models.map((m) => m.model_key);

  const selected = modelKeys.map((key) => {
    const result = byKey.get(key);
    if (!result) {
      throw new Error(`Comparison references model "${key}", which isn't part of the current training run`);
    }
    return result;
  });

  // Re-derived from trainingResults, not comparisonResult's own metric
  // values (docs/plans/pdf-report.md's comparison design decision) -- this
  // keeps the report's numbers sourced from the same place as every other
  // per-model metric it shows.
  const commonMetricNames =
    selected.length === 0
      ? []
      : Object.keys(selected[0].metrics).filter((name) =>
          selected.every((r) => typeof (r.metrics as Record<string, unknown>)[name] === "number"),
        );

  const metricsTable: TableSpec = {
    headers: ["Metric", ...selected.map((r) => r.model_name)],
    rows: commonMetricNames.map((name) => [
      name,
      ...selected.map((r) => formatMetricValue((r.metrics as Record<string, unknown>)[name])),
    ]),
  };

  return { modelKeys, metricsTable };
}

/* -------------------------------------------------------------------------- */
/* Prediction section                                                         */
/* -------------------------------------------------------------------------- */

function buildPredictionSection(
  prediction: NonNullable<CycleState["prediction"]>,
  trainedResults: TrainedModelResponse[],
): ReportModel["prediction"] {
  const { result, inputSummary } = prediction;

  if (!trainedResults.some((r) => r.model_key === result.model_key)) {
    throw new Error(`Prediction references model "${result.model_key}", which isn't part of the current training run`);
  }

  const shown = result.predictions.slice(0, MAX_PREDICTION_ROWS_SHOWN);

  const outputSample: TableSpec = result.probabilities
    ? {
        headers: ["Row", "Prediction", "Probabilities"],
        rows: shown.map((value, i) => [
          i,
          formatPredictionValue(value),
          result.probabilities![i].map((p) => p.toFixed(3)).join(", "),
        ]),
      }
    : {
        headers: ["Row", "Prediction"],
        rows: shown.map((value, i) => [i, formatPredictionValue(value)]),
      };

  const counts = new Map<string, number>();
  for (const value of result.predictions) {
    const key = formatPredictionValue(value);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }

  const predictedValueDistribution: TableSpec = {
    headers: ["Predicted value", "Count"],
    rows: [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([value, count]) => [value, count]),
  };

  return {
    modelKey: result.model_key,
    inputSummary,
    totalRows: result.n_samples,
    outputSample,
    predictedValueDistribution,
  };
}
