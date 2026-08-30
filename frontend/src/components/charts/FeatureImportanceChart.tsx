import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS_TICK, TOOLTIP_STYLE } from "./chartTheme";

interface FeatureImportanceChartProps {
  featureImportance: Record<string, number>;
}

/** Screen 5's "feature importance bar chart where metadata carries it"
 * (frontend.md), rendered wherever `feature_importance` is present,
 * independent of model_type. Sorted descending -- this ranks features
 * within one model's own output, not models against each other, so it
 * isn't the recommendation CLAUDE.md's "no AutoML" rule forbids. */
export function FeatureImportanceChart({ featureImportance }: FeatureImportanceChartProps) {
  const data = Object.entries(featureImportance)
    .sort(([, a], [, b]) => b - a)
    .map(([feature, importance]) => ({ feature, importance }));

  return (
    <div className="h-56">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 16, left: 4, bottom: 4 }}
        >
          <CartesianGrid stroke="var(--color-rule)" horizontal={false} />
          <XAxis type="number" tick={AXIS_TICK} />
          <YAxis type="category" dataKey="feature" tick={AXIS_TICK} width={90} />
          <Tooltip {...TOOLTIP_STYLE} cursor={{ fill: "var(--color-rule)", opacity: 0.3 }} />
          <Bar dataKey="importance" fill="var(--color-ink)" radius={0} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
