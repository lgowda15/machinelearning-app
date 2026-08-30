/** Shared mono-column number formatting (frontend.md's "numeric columns
 * must align in comparison tables") -- used by TrainingScreen's raw result
 * cards, MetricsList (screen 5), and the comparison table (screen 7). */
export function formatMetricValue(value: unknown): string {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  if (value === null || value === undefined) return "—";
  return JSON.stringify(value);
}
