import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AXIS_TICK, TOOLTIP_STYLE } from "./chartTheme";

interface PredictedVsActualChartProps {
  yTrue: number[];
  yPred: number[];
}

/** Screen 5's regressor chart (frontend.md). Points in the regressor's
 * model-type accent (gold); the y=x diagonal (a perfect prediction) in
 * --rule, since it's a reference, not data -- the metrics block above
 * already carries R²/MAE as the key numbers, so this caption just orients
 * the reader to the shape. */
export function PredictedVsActualChart({ yTrue, yPred }: PredictedVsActualChartProps) {
  const data = yTrue.map((actual, i) => ({ x: actual, y: yPred[i] }));
  const allValues = [...yTrue, ...yPred];
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);

  return (
    <div>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
            <CartesianGrid stroke="var(--color-rule)" />
            <XAxis type="number" dataKey="x" name="Actual" tick={AXIS_TICK} domain={[min, max]} />
            <YAxis type="number" dataKey="y" name="Predicted" tick={AXIS_TICK} domain={[min, max]} />
            <Tooltip {...TOOLTIP_STYLE} cursor={{ strokeDasharray: "3 3", stroke: "var(--color-rule)" }} />
            <ReferenceLine
              segment={[
                { x: min, y: min },
                { x: max, y: max },
              ]}
              stroke="var(--color-rule)"
              strokeDasharray="4 4"
            />
            <Scatter data={data} fill="var(--color-type-regress)" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-1 font-mono text-xs text-muted">
        {data.length} test sample{data.length === 1 ? "" : "s"} · dashed line marks a perfect prediction (predicted =
        actual).
      </p>
    </div>
  );
}
