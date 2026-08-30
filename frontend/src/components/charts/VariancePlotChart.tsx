import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS_TICK, TOOLTIP_STYLE } from "./chartTheme";

interface VariancePlotChartProps {
  explainedVarianceRatio: (number | null)[];
}

/** Screen 5's dimensionality-reducer chart (frontend.md): explained
 * variance per component, bars in the reducer's model-type accent
 * (violet). `ratio` can carry `null` entries
 * when the test split has zero variance (metrics.py's generic,
 * non-PCA-specific computation) -- rendered as an empty bar rather than
 * silently treated as zero, so a genuinely undefined value isn't shown as
 * "no variance". */
export function VariancePlotChart({ explainedVarianceRatio }: VariancePlotChartProps) {
  const data = explainedVarianceRatio.map((ratio, i) => ({
    label: `PC${i + 1}`,
    ratio,
  }));
  const definedTotal = explainedVarianceRatio
    .filter((r): r is number => r !== null)
    .reduce((sum, r) => sum + r, 0);
  const hasUndefined = explainedVarianceRatio.some((r) => r === null);

  return (
    <div>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
            <CartesianGrid stroke="var(--color-rule)" vertical={false} />
            <XAxis dataKey="label" tick={AXIS_TICK} />
            <YAxis tick={AXIS_TICK} width={40} />
            <Tooltip {...TOOLTIP_STYLE} cursor={{ fill: "var(--color-rule)", opacity: 0.3 }} />
            <Bar dataKey="ratio" fill="var(--color-type-reduce)" radius={0} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-1 font-mono text-xs text-muted">
        {definedTotal.toFixed(4)} cumulative variance explained
        {hasUndefined ? " (some components undefined -- zero-variance test split)" : ""}
      </p>
    </div>
  );
}
