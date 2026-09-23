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

type TrainedModelResponse =
  components["schemas"]["TrainedModelResponse"];

type TrainResponse = components["schemas"]["TrainResponse"];

interface ResultsScreenProps {
  results: TrainResponse | null;
}

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
    (total, result) =>
      total + (result.training_time_seconds ?? 0),
    0,
  );

  return (
    <ScreenPanel maxWidthClassName={WORKSPACE_WIDTH}>
      {/* Page heading */}
      <div className="mb-5">
        <p className="font-mono text-[10px] uppercase tracking-wider text-muted">
          Training complete
        </p>

        <h1 className="mt-1 text-xl font-semibold tracking-tight text-ink">
          Results
        </h1>

        <p className="mt-1 text-xs text-muted">
          Review the performance and visual output of each trained model.
        </p>
      </div>

      {/* Overview cards */}
      <div className="mb-5 grid grid-cols-1 gap-2 sm:grid-cols-3">
        <OverviewCard
          label="Models trained"
          value={String(results.results.length)}
        />

        <OverviewCard
          label="Test split"
          value={`${Math.round(results.test_size * 100)}%`}
        />

        <OverviewCard
          label="Training time"
          value={`${totalTrainingTime.toFixed(2)}s`}
        />
      </div>

      {/* Model results heading */}
      <div className="mb-3 flex items-end justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-ink">
            Model results
          </h2>

          <p className="mt-0.5 text-[10px] text-muted">
            Performance and visual output for each trained model.
          </p>
        </div>

        <span className="shrink-0 rounded-full bg-ground px-2.5 py-1 font-mono text-[9px] text-muted">
          {results.results.length} trained
        </span>
      </div>

      {/* 2 × 2 model grid */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
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
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-panel border border-rule bg-ground px-4 py-3">
      <p className="text-[9px] font-medium uppercase tracking-wide text-muted">
        {label}
      </p>

      <p className="mt-1 font-mono text-xl font-medium text-ink">
        {value}
      </p>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Individual result card                                                     */
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
      className="
        overflow-hidden
        rounded-panel
        border
        border-rule
        bg-surface
        shadow-sm
      "
    >
      {/* Model header */}
      <div className="border-b border-rule px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2.5">
            <div
              className={`
                flex
                h-7
                w-7
                shrink-0
                items-center
                justify-center
                rounded-panel
                border
                font-mono
                text-[10px]
                ${typeBorder}
                ${typeText}
              `}
            >
              {index + 1}
            </div>

            <div className="min-w-0">
              <h3 className="truncate text-sm font-semibold text-ink">
                {result.model_name}
              </h3>

              <p className="mt-0.5 font-mono text-[9px] text-muted">
                {result.n_features} features
                {result.training_time_seconds != null &&
                  ` · ${result.training_time_seconds.toFixed(3)}s`}
              </p>
            </div>
          </div>

          <span
            className={`
              shrink-0
              rounded-full
              bg-ground
              px-2
              py-1
              font-mono
              text-[8px]
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

      {/* -------------------------------------------------------------- */}
      {/* VISUALISATION — graph first                                    */}
      {/* -------------------------------------------------------------- */}

      <div className="border-b border-rule px-4 py-3">
        <div className="mb-2 flex items-center justify-between">
          <h4 className="text-[9px] font-medium uppercase tracking-wide text-muted">
            Visualisation
          </h4>
        </div>

        <div className="overflow-hidden rounded-panel border border-rule bg-ground p-2">
          <div className="min-h-[180px]">
            <TypeChart result={result} />
          </div>
        </div>
      </div>

      {/* -------------------------------------------------------------- */}
      {/* PERFORMANCE — short section below graph                       */}
      {/* -------------------------------------------------------------- */}

      <div className="border-b border-rule px-4 py-3">
        <div className="mb-2 flex items-center justify-between">
          <h4 className="text-[9px] font-medium uppercase tracking-wide text-muted">
            Performance
          </h4>

          <span className="font-mono text-[8px] text-muted">
            EVALUATION
          </span>
        </div>

        <div className="compact-metrics">
          <MetricsList
            metrics={result.metrics}
            omit={metricsOmitFor(result.model_type)}
          />
        </div>
      </div>

      {/* -------------------------------------------------------------- */}
      {/* FEATURE IMPORTANCE                                             */}
      {/* -------------------------------------------------------------- */}

      {result.feature_importance && (
        <div className="border-b border-rule px-4 py-3">
          <div className="mb-2">
            <h4 className="text-[9px] font-medium uppercase tracking-wide text-muted">
              Feature importance
            </h4>
          </div>

          <div className="overflow-hidden rounded-panel border border-rule bg-ground p-2">
            <FeatureImportanceChart
              featureImportance={result.feature_importance}
            />
          </div>
        </div>
      )}

      {/* -------------------------------------------------------------- */}
      {/* MODEL-SPECIFIC OUTPUT                                          */}
      {/* -------------------------------------------------------------- */}

      {result.visualization_data && (
        <div className="px-4 py-3">
          <div className="mb-2">
            <h4 className="text-[9px] font-medium uppercase tracking-wide text-muted">
              Model-specific output
            </h4>
          </div>

          <div className="overflow-hidden rounded-panel border border-rule bg-ground p-2">
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
          <NoChart
            reason="No cluster scatter data returned for this run."
          />
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
          <NoChart
            reason="No predicted-vs-actual data returned for this run."
          />
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
      <p className="max-w-md text-xs text-muted">
        {reason}
      </p>
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
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 font-mono text-[10px]">
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