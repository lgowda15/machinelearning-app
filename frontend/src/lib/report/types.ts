import type { DataProfileResponse } from "../../hooks/useDataset";
import type { components } from "../../types/api";
import type { LinkageMatrix, ShapValues } from "../../types/visualizationData";

type TrainResponse = components["schemas"]["TrainResponse"];
type ComparisonResponse = components["schemas"]["ComparisonResponse"];
type PredictionResponse = components["schemas"]["PredictionResponse"];
type ColumnSummary = components["schemas"]["ColumnSummary"];
type DataType = components["schemas"]["DataProfileResponse"]["data_type"];

/**
 * Everything a report needs, already resident in App by the time the
 * Download Report button is clicked (docs/plans/pdf-report.md's
 * "Investigation findings" -- no new fetch layer is required). This is
 * assembled fresh from App's lifted state at click time, not itself kept as
 * a single persistent state object.
 *
 * Deviates from the plan's literal `CycleState` sketch in two ways, both to
 * avoid lifting UI state that duplicates what's already resident:
 *  - no separate `dataId` -- it's `profile.data_id`.
 *  - `comparison` carries the raw `ComparisonResponse` instead of a
 *    pre-extracted `{ selectedKeys }`. `buildReportModel` derives the
 *    selected keys (and their order) from `comparisonResult.models`, then
 *    re-derives the metrics table from `trainingResults` -- never from
 *    `comparisonResult`'s own metric values -- exactly as the plan
 *    prescribes ("metrics table re-derived from trainingResults, not
 *    stored separately").
 */
export interface CycleState {
  profile: DataProfileResponse | null;
  testSize: number;
  trainingResults: TrainResponse | null;
  comparisonResult: ComparisonResponse | null;
  prediction: { inputSummary: string; result: PredictionResponse } | null;
}

export interface TableSpec {
  headers: string[];
  rows: (string | number)[][];
}

interface ChartSpecBase {
  title: string;
}

export interface DistributionChartSpec extends ChartSpecBase {
  kind: "distribution";
  column: ColumnSummary;
  targetAccentVar: string;
}

export interface FeatureImportanceChartSpec extends ChartSpecBase {
  kind: "feature-importance";
  featureImportance: Record<string, number>;
}

export interface ClusterScatterChartSpec extends ChartSpecBase {
  kind: "cluster-scatter";
  points: number[][];
  labels: number[];
}

export interface PredictedVsActualChartSpec extends ChartSpecBase {
  kind: "predicted-vs-actual";
  yTrue: number[];
  yPred: number[];
}

export interface VariancePlotChartSpec extends ChartSpecBase {
  kind: "variance-plot";
  explainedVarianceRatio: (number | null)[];
}

export interface ShapValuesChartSpec extends ChartSpecBase {
  kind: "shap-values";
  shapValues: ShapValues;
  featureNames: string[];
  classLabels: string[];
}

export interface DendrogramChartSpec extends ChartSpecBase {
  kind: "dendrogram";
  linkageMatrix: LinkageMatrix;
}

/** Rendered via jspdf-autotable directly, not svg2pdf (plan's render step
 * 3) -- confusion_matrix/labels live inside TrainedModelResponse.metrics
 * (an untyped bag), not as a chart the live app renders as an SVG. */
export interface ConfusionMatrixChartSpec extends ChartSpecBase {
  kind: "confusion-matrix";
  confusionMatrix: number[][];
  labels: string[];
}

/** Rendered via jsPDF's text API directly, not svg2pdf or autotable
 * (plan's "Tree chart" design decision -- no SVG conversion). */
export interface TreeSummaryChartSpec extends ChartSpecBase {
  kind: "tree-summary";
  depth: number;
  nodeCount: number;
  leafCount: number;
  /** Each entry is one root-to-frontier path, already arrow-joined, e.g.
   * "feature_3 <= 0.42 -> feature_1 <= -0.8 -> class 1". */
  rules: string[];
}

export type ChartSpec =
  | DistributionChartSpec
  | FeatureImportanceChartSpec
  | ClusterScatterChartSpec
  | PredictedVsActualChartSpec
  | VariancePlotChartSpec
  | ShapValuesChartSpec
  | DendrogramChartSpec
  | ConfusionMatrixChartSpec
  | TreeSummaryChartSpec;

export interface ReportModelSummary {
  datasetName: string;
  nRows: number;
  nCols: number;
  target: string | null;
  dataType: DataType;
  testSize: number;
  modelDisplayNames: string[];
  generatedAt: string;
}

export interface ReportModelDataset {
  imbalance?: { isImbalanced: boolean; message?: string };
  columns: ColumnSummary[];
  /** 0 when the column list wasn't truncated. */
  truncatedColumnsCount: number;
  distributionCharts: ChartSpec[];
}

export interface ReportModelModel {
  key: string;
  displayName: string;
  modelType: string;
  hyperparameters: Record<string, unknown>;
  metrics: Record<string, unknown>;
  charts: ChartSpec[];
}

export interface ReportModelComparison {
  modelKeys: string[];
  metricsTable: TableSpec;
}

export interface ReportModelPrediction {
  modelKey: string;
  inputSummary: string;
  totalRows: number;
  outputSample: TableSpec;
  predictedValueDistribution: TableSpec;
}

export interface ReportModel {
  summary: ReportModelSummary;
  /** Static section -- fixed pipeline description, not derived from live
   * state (plan's "Preprocessing is backend-owned" design decision). */
  preprocessing: Record<string, never>;
  dataset: ReportModelDataset;
  models: ReportModelModel[];
  /** Omitted entirely (not an empty section) if Compare was never visited. */
  comparison: ReportModelComparison | null;
  /** Omitted entirely (not an empty section) if Predict was never visited. */
  prediction: ReportModelPrediction | null;
}
