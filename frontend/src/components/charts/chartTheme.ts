/** Shared Recharts styling for screen 5's per-type charts -- same tokens
 * and shape DistributionChart already established for screen 2, kept here
 * so ConfusionMatrixChart/ClusterScatterChart/PredictedVsActualChart/
 * VariancePlotChart don't each redefine it. */
export const AXIS_TICK = { fontSize: 10, fontFamily: "var(--font-mono)", fill: "var(--color-muted)" };

export const TOOLTIP_STYLE = {
  contentStyle: {
    background: "var(--color-surface)",
    border: "1px solid var(--color-rule)",
    borderRadius: 4,
    boxShadow: "none",
    fontFamily: "var(--font-mono)",
    fontSize: 12,
  },
};

/** Recharts' built-in Scatter marker shapes, cycled by cluster id so
 * clusters are distinguishable without spending colour on anything but
 * --signal (frontend.md) -- noise (label -1) is marked separately, in
 * --signal, since it's the flagged category, not just another cluster. */
export const CLUSTER_MARKER_SHAPES = [
  "circle",
  "square",
  "triangle",
  "diamond",
  "star",
  "wye",
  "cross",
] as const;

export function markerShapeForCluster(clusterId: number): (typeof CLUSTER_MARKER_SHAPES)[number] {
  const index = clusterId % CLUSTER_MARKER_SHAPES.length;
  return CLUSTER_MARKER_SHAPES[index];
}
