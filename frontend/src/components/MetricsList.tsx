import { Fragment } from "react";
import { formatMetricValue } from "../lib/format";

interface MetricsListProps {
  metrics: Record<string, unknown>;
  // Keys already rendered by the type-appropriate chart (e.g. a
  // classifier's confusion_matrix/labels) -- omitted here so the same
  // array isn't shown twice, once as a chart and once as raw JSON.
  omit?: string[];
}

/** Screen 5's metrics block (frontend.md): mono, labelled, at the top of
 * every result panel, above the type-appropriate chart. */
export function MetricsList({ metrics, omit = [] }: MetricsListProps) {
  const entries = Object.entries(metrics).filter(([key]) => !omit.includes(key));
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-xs">
      {entries.map(([key, value]) => (
        <Fragment key={key}>
          <dt className="text-muted">{key}</dt>
          <dd className="text-ink">{formatMetricValue(value)}</dd>
        </Fragment>
      ))}
    </dl>
  );
}
