import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ScreenPanel, WORKSPACE_WIDTH } from "../ScreenPanel";
import { AXIS_TICK, TOOLTIP_STYLE } from "../charts/chartTheme";
import { formatMetricValue } from "../../lib/format";
import {
  typeBorderClass,
  typeColorVar,
  typeTextClass,
} from "../../lib/modelType";
import { useComparison } from "../../hooks/useComparison";
import type { components } from "../../types/api";

type TrainResponse = components["schemas"]["TrainResponse"];

interface CompareScreenProps {
  trainingResults: TrainResponse | null;
}

const OPACITY_STEPS = [1, 0.7, 0.45, 0.25];

/**
 * Screen 7:
 * - Select trained models
 * - Clearly communicate comparison requirements
 * - Show comparison results as a visual comparison workspace
 * - Models remain in selection order; no automatic ranking
 */
export function CompareScreen({
  trainingResults,
}: CompareScreenProps) {
  const { result, loading, error, compare, reset } = useComparison();
  const [selected, setSelected] = useState<string[]>([]);

  if (!trainingResults || trainingResults.results.length === 0) {
    return (
      <ScreenPanel>
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-ground text-muted">
            —
          </div>

          <h1 className="text-base font-semibold text-ink">
            Nothing to compare yet
          </h1>

          <p className="mt-2 max-w-md text-sm text-muted">
            Train at least two models before comparing their results.
          </p>
        </div>
      </ScreenPanel>
    );
  }

  const byKey = new Map(
    trainingResults.results.map((r) => [r.model_key, r]),
  );

  const selectedModels = selected
    .map((key) => byKey.get(key)!)
    .filter(Boolean);

  const distinctTypes = new Set(
    selectedModels.map((m) => m.model_type),
  );

  const mixedTypes = distinctTypes.size > 1;

  const toggle = (key: string) => {
    reset();

    setSelected((prev) =>
      prev.includes(key)
        ? prev.filter((k) => k !== key)
        : [...prev, key],
    );
  };

  const handleCompare = () => {
    if (selected.length < 2 || mixedTypes) return;

    compare(
      selected.map((model_key) => ({
        training_id: trainingResults.training_id,
        model_key,
      })),
    );
  };

  const canCompare =
    selected.length >= 2 && !mixedTypes && !loading;

  return (
    <ScreenPanel maxWidthClassName={WORKSPACE_WIDTH}>
      {/* Header */}
      <div className="mb-6">
        <p className="font-mono text-xs uppercase tracking-wider text-muted">
          Model analysis
        </p>

        <h1 className="mt-1 text-xl font-semibold text-ink">
          Compare
        </h1>

        <p className="mt-1 text-sm text-muted">
          Select two or more trained models of the same type to compare
          their performance.
        </p>
      </div>

      {/* Model selection */}
      <section className="mb-6 rounded-panel border border-rule bg-ground p-5">
        <div className="mb-4 flex items-center justify-between gap-4">
          <div>
            <h2 className="text-sm font-semibold text-ink">
              Select models
            </h2>

            <p className="mt-1 text-xs text-muted">
              {selected.length === 0
                ? "Choose at least two models."
                : `${selected.length} model${selected.length === 1 ? "" : "s"} selected`}
            </p>
          </div>

          {selected.length > 0 && (
            <button
              type="button"
              onClick={() => {
                reset();
                setSelected([]);
              }}
              className="cursor-pointer text-xs font-medium text-muted transition-colors hover:text-ink"
            >
              Clear selection
            </button>
          )}
        </div>

        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
          {trainingResults.results.map((model) => {
            const isSelected = selected.includes(model.model_key);

            return (
              <button
                key={model.model_key}
                type="button"
                aria-pressed={isSelected}
                onClick={() => toggle(model.model_key)}
                className={`
                  flex
                  min-h-20
                  items-center
                  gap-3
                  rounded-panel
                  border
                  bg-surface
                  px-4
                  py-3
                  text-left
                  transition-all
                  duration-150
                  ${
                    isSelected
                      ? `${typeBorderClass(model.model_type)} shadow-sm`
                      : "cursor-pointer border-rule hover:border-ink/40 hover:shadow-sm"
                  }
                `}
              >
                {/* Selection indicator */}
                <span
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
                    text-xs
                    ${
                      isSelected
                        ? `${typeBorderClass(model.model_type)} ${typeTextClass(model.model_type)}`
                        : "border-rule text-muted"
                    }
                  `}
                >
                  {isSelected ? "✓" : ""}
                </span>

                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-ink">
                    {model.model_name}
                  </p>

                  <p
                    className={`mt-1 font-mono text-[10px] uppercase tracking-wide ${
                      isSelected
                        ? typeTextClass(model.model_type)
                        : "text-muted"
                    }`}
                  >
                    {model.model_type}
                  </p>
                </div>
              </button>
            );
          })}
        </div>

        {/* Selection guidance */}
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-rule pt-4">
          <p className="text-xs text-muted">
            Models are compared in the order you select them.
          </p>

          <button
            type="button"
            onClick={handleCompare}
            disabled={!canCompare}
            className={`
              rounded-panel
              border
              px-5
              py-2
              text-sm
              font-medium
              transition-all
              duration-150
              ${
                canCompare
                  ? "cursor-pointer border-signal bg-signal text-surface hover:opacity-90 active:scale-[0.98]"
                  : "cursor-not-allowed border-rule bg-surface text-muted opacity-60"
              }
            `}
          >
            {loading ? "Comparing…" : "Compare selected models"}
          </button>
        </div>
      </section>

      {/* Mixed model types */}
      {mixedTypes && (
        <div className="mb-6 rounded-panel border border-rule bg-ground px-4 py-3">
          <p className="text-sm font-medium text-ink">
            Models cannot be compared together
          </p>

          <p className="mt-1 text-xs leading-relaxed text-muted">
            Select models of the same type. Your current selection contains{" "}
            {[...distinctTypes].join(", ")}.
          </p>
        </div>
      )}

      {/* Too few models */}
      {selected.length === 1 && !mixedTypes && (
        <p className="mb-6 text-xs text-muted">
          Select at least one more model to enable comparison.
        </p>
      )}

      {/* Error */}
      {error && (
        <div className="mb-6 rounded-panel border border-rule bg-ground px-4 py-3">
          <p className="text-sm font-medium text-ink">
            Comparison failed
          </p>

          <p className="mt-1 text-xs text-muted">
            {error.message}
          </p>
        </div>
      )}

      {/* Results */}
      {result && <ComparisonResult result={result} />}
    </ScreenPanel>
  );
}

/* -------------------------------------------------------------------------- */
/* Comparison result                                                         */
/* -------------------------------------------------------------------------- */

function ComparisonResult({
  result,
}: {
  result: components["schemas"]["ComparisonResponse"];
}) {
  if (result.common_metrics.length === 0) {
    return (
      <div className="rounded-panel border border-rule bg-ground px-4 py-6 text-center">
        <p className="text-sm font-medium text-ink">
          No common metrics available
        </p>

        <p className="mt-1 text-xs text-muted">
          These models do not share any metrics that can be compared.
        </p>
      </div>
    );
  }

  const numericMetrics = result.common_metrics.filter((metric) =>
    result.models.every(
      (m) => typeof m.metrics[metric] === "number",
    ),
  );

  const chartData = numericMetrics.map((metric) => {
    const row: Record<string, string | number> = { metric };

    result.models.forEach((m) => {
      row[m.model_key] = m.metrics[metric] as number;
    });

    return row;
  });

  return (
    <div className="space-y-6">
      {/* Result heading */}
      <div className="border-b border-rule pb-4">
        <p className="font-mono text-xs uppercase tracking-wider text-muted">
          Comparison complete
        </p>

        <h2 className="mt-1 text-lg font-semibold text-ink">
          Model comparison
        </h2>

        <p className="mt-1 text-sm text-muted">
          Metrics are shown side-by-side in your original selection order.
        </p>
      </div>

      {/* Model summary cards */}
      <div
        className={`grid gap-3 ${
          result.models.length === 2
            ? "md:grid-cols-2"
            : "md:grid-cols-2 xl:grid-cols-3"
        }`}
      >
        {result.models.map((model, index) => (
          <div
            key={model.model_key}
            className={`
              rounded-panel
              border
              border-rule
              border-l-4
              bg-surface
              px-4
              py-4
              ${typeBorderClass(model.model_type)}
            `}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex min-w-0 items-start gap-3">
                <span className="font-mono text-xs text-muted">
                  {String(index + 1).padStart(2, "0")}
                </span>

                <div className="min-w-0">
                  <h3 className="truncate text-sm font-semibold text-ink">
                    {model.model_name}
                  </h3>

                  <p
                    className={`mt-1 font-mono text-[10px] uppercase tracking-wide ${typeTextClass(
                      model.model_type,
                    )}`}
                  >
                    {model.model_type}
                  </p>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Metrics table */}
      <section>
        <div className="mb-3">
          <h3 className="text-xs font-medium uppercase tracking-wide text-muted">
            Metric comparison
          </h3>
        </div>

        <div className="overflow-x-auto rounded-panel border border-rule bg-surface">
          <table className="w-full min-w-[600px] text-left text-sm">
            <thead>
              <tr className="border-b border-rule bg-ground">
                <th className="px-4 py-3 text-xs font-medium uppercase tracking-wide text-muted">
                  Metric
                </th>

                {result.models.map((m) => (
                  <th
                    key={m.model_key}
                    className={`px-4 py-3 font-mono text-xs font-medium ${typeTextClass(
                      m.model_type,
                    )}`}
                  >
                    {m.model_name}
                  </th>
                ))}
              </tr>
            </thead>

            <tbody>
              {result.common_metrics.map((metric) => (
                <tr
                  key={metric}
                  className="border-b border-rule last:border-b-0"
                >
                  <td className="px-4 py-3 text-xs font-medium text-muted">
                    {metric}
                  </td>

                  {result.models.map((m) => (
                    <td
                      key={m.model_key}
                      className="px-4 py-3 font-mono text-sm text-ink"
                    >
                      {formatMetricValue(m.metrics[metric])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Visual comparison */}
      {chartData.length > 0 && (
        <section>
          <div className="mb-3">
            <h3 className="text-xs font-medium uppercase tracking-wide text-muted">
              Visual comparison
            </h3>

            <p className="mt-1 text-xs text-muted">
              Numeric metrics shown side-by-side. Models retain their
              selection order.
            </p>
          </div>

          <div className="h-80 rounded-panel border border-rule bg-ground p-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData}
                margin={{
                  top: 8,
                  right: 16,
                  left: 0,
                  bottom: 8,
                }}
              >
                <CartesianGrid
                  stroke="var(--color-rule)"
                  vertical={false}
                />

                <XAxis
                  dataKey="metric"
                  tick={AXIS_TICK}
                />

                <YAxis
                  tick={AXIS_TICK}
                  width={48}
                />

                <Tooltip
                  {...TOOLTIP_STYLE}
                  cursor={{
                    fill: "var(--color-rule)",
                    opacity: 0.15,
                  }}
                />

                <Legend
                  wrapperStyle={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                  }}
                />

                {result.models.map((m, index) => (
                  <Bar
                    key={m.model_key}
                    dataKey={m.model_key}
                    name={m.model_name}
                    fill={typeColorVar(m.model_type)}
                    fillOpacity={
                      OPACITY_STEPS[index] ?? 0.25
                    }
                    radius={2}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}
    </div>
  );
}