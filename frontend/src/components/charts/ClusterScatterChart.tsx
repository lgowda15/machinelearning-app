import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AXIS_TICK, TOOLTIP_STYLE, markerShapeForCluster } from "./chartTheme";

interface ClusterScatterChartProps {
  points: number[][]; // [x, y] per test sample -- app.core.metrics._cluster_scatter_points
  labels: number[]; // predicted cluster id per sample; -1 is noise (never relabelled, CLAUDE.md)
}

const NOISE_LABEL = -1;

/** Screen 5's clusterer chart (frontend.md). `points` is a 2D projection
 * for display only (PCA when there were more than two features to
 * project, computed server-side so this component stays a thin renderer --
 * see metrics.py's docstring for why). Clusters are told apart by marker
 * shape, not colour, so every ordinary cluster shares one fill: the
 * clusterer's model-type accent (frontend.md's colour coding), the same
 * teal-green as this model's card in Model Selection. --signal stays
 * reserved for noise, the one flagged category -- --signal still marks
 * "the thing in focus" generally, it's just no longer this chart's
 * type colour. */
export function ClusterScatterChart({ points, labels }: ClusterScatterChartProps) {
  const byLabel = new Map<number, { x: number; y: number }[]>();
  points.forEach((point, i) => {
    const label = labels[i];
    const series = byLabel.get(label) ?? [];
    series.push({ x: point[0], y: point[1] });
    byLabel.set(label, series);
  });

  const clusterIds = [...byLabel.keys()].filter((id) => id !== NOISE_LABEL).sort((a, b) => a - b);
  const noisePoints = byLabel.get(NOISE_LABEL);
  const noiseCount = noisePoints?.length ?? 0;

  return (
    <div>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
            <CartesianGrid stroke="var(--color-rule)" />
            <XAxis type="number" dataKey="x" name="Component 1" tick={AXIS_TICK} />
            <YAxis type="number" dataKey="y" name="Component 2" tick={AXIS_TICK} />
            <Tooltip {...TOOLTIP_STYLE} cursor={{ strokeDasharray: "3 3", stroke: "var(--color-rule)" }} />
            <Legend wrapperStyle={{ fontFamily: "var(--font-mono)", fontSize: 11 }} />
            {clusterIds.map((clusterId, index) => (
              <Scatter
                key={clusterId}
                name={`cluster ${clusterId}`}
                data={byLabel.get(clusterId)}
                fill="var(--color-type-cluster)"
                shape={markerShapeForCluster(index)}
              />
            ))}
            {noisePoints && (
              <Scatter name="noise" data={noisePoints} fill="var(--color-signal)" shape="cross" />
            )}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-1 font-mono text-xs text-muted">
        {clusterIds.length} cluster{clusterIds.length === 1 ? "" : "s"} · {noiseCount} noise point
        {noiseCount === 1 ? "" : "s"}
      </p>
    </div>
  );
}
