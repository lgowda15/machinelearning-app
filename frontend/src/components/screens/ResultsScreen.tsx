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
import type {
  LinkageMatrix,
  ShapValues,
  TreeStructurePayload,
} from "../../types/visualizationData";

type TrainedModelResponse = components["schemas"]["TrainedModelResponse"];
type TrainResponse = components["schemas"]["TrainResponse"];

interface ResultsScreenProps {
  results: TrainResponse | null;
}

/**
 * Screen 5:
 * - Overview of the completed training run
 * - One visual result card per trained model
 * - Key metrics shown prominently
 * - Type-specific visualisation
 * - Feature importance / model-specific visualisations where available
 */
export function ResultsScreen({ results }: ResultsScreenProps) {
  if (!results || results.results.length === 0) {
    return (
      <ScreenPanel>
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-ground text-muted">
            —
          </div>

          <h1 className="text-base font-semibold text-ink">
            No results yet
          </h1>

          <p className="mt-2 max-w-md text-sm text-muted">
            Train at least one model to see its metrics and visualisations
            here.
          </p>
        </div>
      </ScreenPanel>
    );
  }

  const totalTrainingTime = results.results.reduce(
    (total, result) => total + (result.training_time_seconds ?? 0),
    0,
  );

  return (
    <ScreenPanel maxWidthClassName={WORKSPACE_WIDTH}>
      {/* Page heading */}
      <div className="mb-6">
        <p className="font-mono text-xs uppercase tracking-wider text-muted">
          Training complete
        </p>

        <h1 className="mt-1 text-xl font-semibold text-ink">
          Results
        </h1>

        <p className="mt-1 text-sm text-muted">
          Review the performance and visual output of each trained model.
        </p>
      </div>

      {/* Overview cards */}
      <div className="mb-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <OverviewCard
          label="Models trained"
          value={String(results.results.length)}
          detail="Models completed successfully"
        />

        <OverviewCard
          label="Test split"
          value={`${Math.round(results.test_size * 100)}%`}
          detail="Data used for evaluation"
        />

        <OverviewCard
          label="Training time"
          value={`${totalTrainingTime.toFixed(2)}s`}
          detail="Total model training time"
        />
      </div>

      {/* Model results */}
      <div className="space-y-6">
        {results.results.map((result, index) => (
          <ResultPanel
            key={result.model_key}
            result={result}
            index={index}
          />
        ))}
      </div>
    </ScreenPanel>
  );
}

/* -------------------------------------------------------------------------- */
/* Overview card                                                              */
/* -------------------------------------------------------------------------- */

function OverviewCard({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="rounded-panel border border-rule bg-ground px-4 py-4">
      <p className="text-xs font-medium uppercase tracking-wide text-muted">
        {label}
      </p>

      <p className="mt-2 font-mono text-2xl font-medium text-ink">
        {value}
      </p>

      <p className="mt-1 text-xs text-muted">
        {detail}
      </p>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Individual result                                                          */
/* -------------------------------------------------------------------------- */

function ResultPanel({
  result,
  index,
}: {
  result: TrainedModelResponse;
  index: number;
}) {
  const typeText = typeTextClass(result.model_type);
  const typeBorder = typeBorderClass(result.model_type);

  return (
    <section
      className={`
        overflow-hidden
        rounded-panel
        border
        border-rule
        bg-surface
        shadow-sm
      `}
    >
      {/* Model header */}
      <div className="border-b border-rule px-5 py-4">
        <div className="flex items-start justify-between gap-4">
          <div className="flex min-w-0 items-start gap-3">
            <div
              className={`
                flex
                h-8
                w-8
                shrink-0
                items-center
                justify-center
                rounded-panel
                border
                ${typeBorder}
                font-mono
                text-xs
                ${typeText}
              `}
            >
              {index + 1}
            </div>

            <div className="min-w-0">
              <h2 className="text-base font-semibold text-ink">
                {result.model_name}
              </h2>

              <p className="mt-1 font-mono text-xs text-muted">
                {result.n_features} features
                {result.training_time_seconds != null &&
                  ` · ${result.training_time_seconds.toFixed(3)}s training time`}
              </p>
            </div>
          </div>

          <span
            className={`
              shrink-0
              rounded-full
              bg-ground
              px-2.5
              py-1
              font-mono
              text-[10px]
              font-medium
              uppercase
              tracking-wide
              ${typeText}
            `}
          >
            {result.model_type}
          </span>
        </div>
      </div>

      {/* Metrics */}
      <div className="border-b border-rule px-5 py-5">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-xs font-medium uppercase tracking-wide text-muted">
            Performance
          </h3>

          <span className="font-mono text-[10px] text-muted">
            EVALUATION
          </span>
        </div>

        <MetricsList
          metrics={result.metrics}
          omit={metricsOmitFor(result.model_type)}
        />
      </div>

      {/* Main visualisation */}
      <div className="border-b border-rule px-5 py-5">
        <div className="mb-3">
          <h3 className="text-xs font-medium uppercase tracking-wide text-muted">
            Visualisation
          </h3>
        </div>

        <div className="overflow-hidden rounded-panel border border-rule bg-ground p-3">
          <TypeChart result={result} />
        </div>
      </div>

      {/* Feature importance */}
      {result.feature_importance && (
        <div className="border-b border-rule px-5 py-5">
          <div className="mb-3">
            <h3 className="text-xs font-medium uppercase tracking-wide text-muted">
              Feature importance
            </h3>

            <p className="mt-1 text-xs text-muted">
              Relative contribution of each feature to the model.
            </p>
          </div>

          <div className="overflow-hidden rounded-panel border border-rule bg-ground p-3">
            <FeatureImportanceChart
              featureImportance={result.feature_importance}
            />
          </div>
        </div>
      )}

      {/* Model-specific visualisation */}
      {result.visualization_data && (
        <div className="px-5 py-5">
          <div className="mb-3">
            <h3 className="text-xs font-medium uppercase tracking-wide text-muted">
              Model-specific output
            </h3>
          </div>

          <div className="overflow-hidden rounded-panel border border-rule bg-ground p-3">
            <ModelSpecificVisual
              result={result}
              data={result.visualization_data}
            />
          </div>
        </div>
      )}
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/* Model-specific visualisation                                               */
/* -------------------------------------------------------------------------- */

/**
 * Dispatches visualization_data to the chart built for its specific shape.
 */
function ModelSpecificVisual({
  result,
  data,
}: {
  result: TrainedModelResponse;
  data: Record<string, unknown>;
}) {
  if ("tree_structure" in data) {
    return (
      <TreeStructureChart
        data={data.tree_structure as TreeStructurePayload}
      />
    );
  }

  if ("shap_values" in data) {
    const metrics = result.metrics as Record<string, unknown>;

    const featureNames = Array.from(
      { length: result.n_features },
      (_, i) => `feature_${i}`,
    );

    const classLabels =
      (metrics.labels as string[] | undefined) ?? [];

    return (
      <ShapValuesChart
        shapValues={data.shap_values as ShapValues}
        featureNames={featureNames}
        classLabels={classLabels}
      />
    );
  }

  if ("linkage_matrix" in data) {
    return (
      <DendrogramChart
        linkageMatrix={data.linkage_matrix as LinkageMatrix}
      />
    );
  }

  return <ModelVisualizationDump data={data} />;
}

/* -------------------------------------------------------------------------- */
/* Metric helpers                                                             */
/* -------------------------------------------------------------------------- */

function metricsOmitFor(modelType: string): string[] {
  return modelType === "classifier"
    ? ["confusion_matrix", "labels"]
    : [];
}

/* -------------------------------------------------------------------------- */
/* Main chart                                                                 */
/* -------------------------------------------------------------------------- */

function TypeChart({
  result,
}: {
  result: TrainedModelResponse;
}) {
  const metrics = result.metrics as Record<string, unknown>;

  switch (result.model_type) {
    case "classifier":
      return (
        <ConfusionMatrixChart
          confusionMatrix={
            metrics.confusion_matrix as number[][]
          }
          labels={metrics.labels as string[]}
        />
      );

    case "clusterer": {
      const plotData = result.plot_data as {
        points: number[][];
        labels: number[];
      } | null;

      if (!plotData) {
        return (
          <NoChart reason="No cluster scatter data returned for this run." />
        );
      }

      return (
        <ClusterScatterChart
          points={plotData.points}
          labels={plotData.labels}
        />
      );
    }

    case "regressor": {
      const plotData = result.plot_data as {
        y_true: number[];
        y_pred: number[];
      } | null;

      if (!plotData) {
        return (
          <NoChart reason="No predicted-vs-actual data returned for this run." />
        );
      }

      return (
        <PredictedVsActualChart
          yTrue={plotData.y_true}
          yPred={plotData.y_pred}
        />
      );
    }

    case "dimensionality_reducer":
      return (
        <VariancePlotChart
          explainedVarianceRatio={
            metrics.explained_variance_ratio as (number | null)[]
          }
        />
      );

    default:
      return (
        <NoChart
          reason={`Unrecognised model type '${result.model_type}'.`}
        />
      );
  }
}

/* -------------------------------------------------------------------------- */
/* Empty chart state                                                          */
/* -------------------------------------------------------------------------- */

function NoChart({ reason }: { reason: string }) {
  return (
    <div className="flex min-h-32 items-center justify-center text-center">
      <p className="max-w-md text-sm text-muted">{reason}</p>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Fallback visualisation                                                     */
/* -------------------------------------------------------------------------- */

function ModelVisualizationDump({
  data,
}: {
  data: Record<string, unknown>;
}) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 font-mono text-xs">
      {Object.entries(data).map(([key, value]) => (
        <Fragment key={key}>
          <dt className="text-muted">{key}</dt>

          <dd className="break-all text-ink">
            {Array.isArray(value)
              ? `[${value.length} values]`
              : JSON.stringify(value)}
          </dd>
        </Fragment>
      ))}
    </dl>
  );
}