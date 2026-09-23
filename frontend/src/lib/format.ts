/** Shared mono-column number formatting (frontend.md's "numeric columns
 * must align in comparison tables") -- used by TrainingScreen's raw result
 * cards, MetricsList (screen 5), and the comparison table (screen 7). */
export function formatMetricValue(value: unknown): string {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  if (value === null || value === undefined) return "—";
  return JSON.stringify(value);
}

/** Formats one row of a prediction result (Screen 6 and the PDF report).
 * A dimensionality reducer's predict returns a 2D row (CLAUDE.md's "known
 * contract exception"), hence the array branch. */
export function formatPredictionValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value
      .map((v) => (typeof v === "number" ? v.toFixed(4) : String(v)))
      .join(", ");
  }

  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }

  return String(value);
}
