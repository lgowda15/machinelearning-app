import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ScreenPanel, WORKSPACE_WIDTH } from "../ScreenPanel";
import { AXIS_TICK, TOOLTIP_STYLE } from "../charts/chartTheme";
import { formatMetricValue } from "../../lib/format";
import { typeBorderClass, typeColorVar, typeTextClass } from "../../lib/modelType";
import { useComparison } from "../../hooks/useComparison";
import type { components } from "../../types/api";

type TrainResponse = components["schemas"]["TrainResponse"];

interface CompareScreenProps {
  trainingResults: TrainResponse | null;
}

// Compare only ever holds one model_type (mixed selections are blocked
// below), so every bar shares that type's accent colour (frontend.md's
// colour coding) -- individual models are told apart by decreasing
// opacity in selection order, first-selected fully opaque, rather than a
// per-model palette.
const OPACITY_STEPS = [1, 0.7, 0.45, 0.25];

/** Screen 7 (frontend.md): metrics table (models as columns, metrics as
 * rows), then a grouped bar chart, ordered by selection order -- never by
 * score, which would read as ranking (CLAUDE.md's "no AutoML"). Only
 * models of the same model_type are comparable; a mixed selection shows an
 * explanation instead of an empty table or a failed request. */
export function CompareScreen({ trainingResults }: CompareScreenProps) {
  const { result, loading, error, compare, reset } = useComparison();
  const [selected, setSelected] = useState<string[]>([]);

  if (!trainingResults || trainingResults.results.length === 0) {
    return (
      <ScreenPanel>
        <p className="text-sm text-muted">Train at least one model before comparing results.</p>
      </ScreenPanel>
    );
  }

  const byKey = new Map(trainingResults.results.map((r) => [r.model_key, r]));
  const selectedModels = selected.map((key) => byKey.get(key)!).filter(Boolean);
  const distinctTypes = new Set(selectedModels.map((m) => m.model_type));
  const mixedTypes = distinctTypes.size > 1;

  const toggle = (key: string) => {
    reset();
    setSelected((prev) => (prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]));
  };

  const handleCompare = () => {
    if (selected.length < 2 || mixedTypes) return;
    compare(selected.map((model_key) => ({ training_id: trainingResults.training_id, model_key })));
  };

  return (
    <ScreenPanel maxWidthClassName={WORKSPACE_WIDTH}>
      <h1 className="mb-1 text-lg font-medium text-ink">Compare</h1>
      <p className="mb-4 text-sm text-muted">Select two or more trained models of the same type to compare.</p>

      <ul className="mb-4 flex flex-wrap gap-2">
        {trainingResults.results.map((model) => {
          const isSelected = selected.includes(model.model_key);
          return (
            <li key={model.model_key}>
              <button
                type="button"
                aria-pressed={isSelected}
                onClick={() => toggle(model.model_key)}
                className={
                  "rounded-panel border px-3 py-2 text-left text-sm " +
                  (isSelected
                    ? typeBorderClass(model.model_type) + " " + typeTextClass(model.model_type)
                    : "border-rule text-ink hover:border-ink")
                }
              >
                {model.model_name}
                <span className="ml-2 font-mono text-xs text-muted">{model.model_type}</span>
              </button>
            </li>
          );
        })}
      </ul>

      {mixedTypes && (
        <p className="mb-4 text-sm text-ink">
          These models aren't comparable: {[...distinctTypes].join(", ")} are different model types. Select
          models of one type.
        </p>
      )}

      <button
        type="button"
        onClick={handleCompare}
        disabled={selected.length < 2 || mixedTypes || loading}
        className="mb-6 rounded-panel border border-signal bg-signal px-4 py-2 text-sm font-medium text-surface disabled:cursor-not-allowed disabled:opacity-50"
      >
        {loading ? "Comparing…" : "Compare selected models"}
      </button>

      {error && <p className="mb-4 text-sm text-ink">Comparison failed: {error.message}</p>}

      {result && <ComparisonResult result={result} />}
    </ScreenPanel>
  );
}

function ComparisonResult({ result }: { result: components["schemas"]["ComparisonResponse"] }) {
  if (result.common_metrics.length === 0) {
    return <p className="text-sm text-muted">These models share no common metrics to compare.</p>;
  }

  const numericMetrics = result.common_metrics.filter((metric) =>
    result.models.every((m) => typeof m.metrics[metric] === "number"),
  );
  const chartData = numericMetrics.map((metric) => {
    const row: Record<string, string | number> = { metric };
    result.models.forEach((m) => {
      row[m.model_key] = m.metrics[metric] as number;
    });
    return row;
  });

  return (
    <div className="flex flex-col gap-6">
      <div className="overflow-x-auto rounded-panel border border-rule bg-surface">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-rule text-xs uppercase tracking-wide text-muted">
              <th className="px-3 py-2 font-medium">Metric</th>
              {result.models.map((m) => (
                <th
                  key={m.model_key}
                  className={"px-3 py-2 font-mono font-medium normal-case " + typeTextClass(m.model_type)}
                >
                  {m.model_name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {result.common_metrics.map((metric) => (
              <tr key={metric} className="border-b border-rule last:border-b-0">
                <td className="px-3 py-2 text-muted">{metric}</td>
                {result.models.map((m) => (
                  <td key={m.model_key} className="px-3 py-2 font-mono text-ink">
                    {formatMetricValue(m.metrics[metric])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {chartData.length > 0 && (
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
              <CartesianGrid stroke="var(--color-rule)" vertical={false} />
              <XAxis dataKey="metric" tick={AXIS_TICK} />
              <YAxis tick={AXIS_TICK} width={48} />
              <Tooltip {...TOOLTIP_STYLE} cursor={{ fill: "var(--color-rule)", opacity: 0.15 }} />
              <Legend wrapperStyle={{ fontFamily: "var(--font-mono)", fontSize: 11 }} />
              {result.models.map((m, index) => (
                <Bar
                  key={m.model_key}
                  dataKey={m.model_key}
                  name={m.model_name}
                  fill={typeColorVar(m.model_type)}
                  fillOpacity={OPACITY_STEPS[index] ?? 0.25}
                  radius={0}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
