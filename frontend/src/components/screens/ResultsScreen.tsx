import { Fragment } from "react";
import { ClusterScatterChart } from "../charts/ClusterScatterChart";
import { ConfusionMatrixChart } from "../charts/ConfusionMatrixChart";
import { DendrogramChart } from "../charts/DendrogramChart";
import { FeatureImportanceChart } from "../charts/FeatureImportanceChart";
import { PredictedVsActualChart } from "../charts/PredictedVsActualChart";
import { ShapValuesChart } from "../charts/ShapValuesChart";
import { TreeStructureChart } from "../charts/TreeStructureChart";
import { VariancePlotChart } from "../charts/VariancePlotChart";
import { MetricsList } from "../MetricsList";
import { ScreenPanel, WORKSPACE_WIDTH } from "../ScreenPanel";
import { typeBorderClass, typeTextClass } from "../../lib/modelType";
import type { components } from "../../types/api";
import type { LinkageMatrix, ShapValues, TreeStructurePayload } from "../../types/visualizationData";

type TrainedModelResponse = components["schemas"]["TrainedModelResponse"];
type TrainResponse = components["schemas"]["TrainResponse"];

interface ResultsScreenProps {
  results: TrainResponse | null;
}

/** Screen 5 (frontend.md): one panel per trained model, stacked -- metrics
 * block, then the type-appropriate chart, feature importance where
 * present, model-specific visual last where present. */
export function ResultsScreen({ results }: ResultsScreenProps) {
  if (!results || results.results.length === 0) {
    return (
      <ScreenPanel>
        <p className="text-sm text-muted">Train at least one model to see results.</p>
      </ScreenPanel>
    );
  }

  return (
    <ScreenPanel maxWidthClassName={WORKSPACE_WIDTH}>
      <h1 className="mb-4 text-lg font-medium text-ink">Results</h1>
      <div className="flex flex-col gap-6">
        {results.results.map((result) => (
          <ResultPanel key={result.model_key} result={result} />
        ))}
      </div>
    </ScreenPanel>
  );
}

function ResultPanel({ result }: { result: TrainedModelResponse }) {
  // Model-type colour coding (frontend.md): a left accent strip plus the
  // type label itself, matching the same model's card in Model Selection
  // and, once shared with another model, its column in Compare.
  return (
    <section className={"rounded-panel border border-l-4 p-4 " + typeBorderClass(result.model_type)}>
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-ink">{result.model_name}</h2>
        <span className={"font-mono text-xs uppercase " + typeTextClass(result.model_type)}>
          {result.model_type}
        </span>
      </div>

      <p className="mt-1 font-mono text-xs text-muted">
        {result.n_features} features
        {result.training_time_seconds != null && ` · trained in ${result.training_time_seconds.toFixed(3)}s`}
      </p>

      <div className="mt-3">
        <MetricsList metrics={result.metrics} omit={metricsOmitFor(result.model_type)} />
      </div>

      <div className="mt-4">
        <TypeChart result={result} />
      </div>

      {result.feature_importance && (
        <div className="mt-4">
          <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted">Feature importance</p>
          <FeatureImportanceChart featureImportance={result.feature_importance} />
        </div>
      )}

      {result.visualization_data && (
        <div className="mt-4">
          <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted">Model-provided data</p>
          <ModelSpecificVisual result={result} data={result.visualization_data} />
        </div>
      )}
    </section>
  );
}

/** Dispatches `visualization_data` to the chart built for its specific
 * shape (DATA_FLOW_GUIDE.md §5.3: "whatever JSON-safe structure they
 * return is handed to a matching frontend chart component we build for
 * that specific shape") -- group_01's `tree_structure`, group_02's
 * `shap_values`, group_09's `linkage_matrix`. Anything else (group_15's
 * `explained_variance_ratio`, or a future group) falls back to the raw
 * key/value dump this screen already had. */
function ModelSpecificVisual({
  result,
  data,
}: {
  result: TrainedModelResponse;
  data: Record<string, unknown>;
}) {
  if ("tree_structure" in data) {
    return <TreeStructureChart data={data.tree_structure as TreeStructurePayload} />;
  }
  if ("shap_values" in data) {
    const metrics = result.metrics as Record<string, unknown>;
    const featureNames = Array.from({ length: result.n_features }, (_, i) => `feature_${i}`);
    const classLabels = (metrics.labels as string[] | undefined) ?? [];
    return (
      <ShapValuesChart
        shapValues={data.shap_values as ShapValues}
        featureNames={featureNames}
        classLabels={classLabels}
      />
    );
  }
  if ("linkage_matrix" in data) {
    return <DendrogramChart linkageMatrix={data.linkage_matrix as LinkageMatrix} />;
  }
  return <ModelVisualizationDump data={data} />;
}

// The chart already renders these keys visually -- kept out of the raw
// metrics list so a confusion matrix isn't also dumped as a JSON array.
function metricsOmitFor(modelType: string): string[] {
  return modelType === "classifier" ? ["confusion_matrix", "labels"] : [];
}

/** The one place model_type is a rendering switch, not just a metrics
 * switch (CLAUDE.md "model_type is a switch, not documentation") -- all
 * four branches must render, per BUILD_SESSIONS.md Session 7. */
function TypeChart({ result }: { result: TrainedModelResponse }) {
  const metrics = result.metrics as Record<string, unknown>;

  switch (result.model_type) {
    case "classifier":
      return (
        <ConfusionMatrixChart
          confusionMatrix={metrics.confusion_matrix as number[][]}
          labels={metrics.labels as string[]}
        />
      );
    case "clusterer": {
      const plotData = result.plot_data as { points: number[][]; labels: number[] } | null;
      if (!plotData) return <NoChart reason="No cluster scatter data returned for this run." />;
      return <ClusterScatterChart points={plotData.points} labels={plotData.labels} />;
    }
    case "regressor": {
      const plotData = result.plot_data as { y_true: number[]; y_pred: number[] } | null;
      if (!plotData) return <NoChart reason="No predicted-vs-actual data returned for this run." />;
      return <PredictedVsActualChart yTrue={plotData.y_true} yPred={plotData.y_pred} />;
    }
    case "dimensionality_reducer":
      return (
        <VariancePlotChart explainedVarianceRatio={metrics.explained_variance_ratio as (number | null)[]} />
      );
    default:
      // model_type is validated server-side (app.core.metrics raises on an
      // unrecognised value); this is unreachable for a well-formed response.
      return <NoChart reason={`Unrecognised model type '${result.model_type}'.`} />;
  }
}

function NoChart({ reason }: { reason: string }) {
  return <p className="text-sm text-muted">{reason}</p>;
}

function ModelVisualizationDump({ data }: { data: Record<string, unknown> }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-xs">
      {Object.entries(data).map(([key, value]) => (
        <Fragment key={key}>
          <dt className="text-muted">{key}</dt>
          <dd className="break-all text-ink">
            {Array.isArray(value) ? `[${value.length} values]` : JSON.stringify(value)}
          </dd>
        </Fragment>
      ))}
    </dl>
  );
}
