import { bucketsForColumn } from "../lib/distribution";
import type { components } from "../types/api";

type ColumnSummary = components["schemas"]["ColumnSummary"];

interface SparklineProps {
  column: ColumnSummary;
  color?: string;
}

const WIDTH = 56;
const HEIGHT = 18;

/**
 * Tiny inline bar sparkline for the EDA screen's column rail -- the same
 * bucket shaping as the main DistributionChart (lib/distribution.ts), at a
 * glance rather than with axes/labels/tooltip.
 */
export function Sparkline({ column, color = "var(--color-muted)" }: SparklineProps) {
  const buckets = bucketsForColumn(column);

  if (buckets.length === 0) {
    return <svg width={WIDTH} height={HEIGHT} role="presentation" aria-hidden="true" />;
  }

  const max = Math.max(...buckets.map((b) => b.count), 1);
  const barWidth = WIDTH / buckets.length;

  return (
    <svg width={WIDTH} height={HEIGHT} role="presentation" aria-hidden="true">
      {buckets.map((bucket, i) => {
        const barHeight = Math.max(1, (bucket.count / max) * HEIGHT);
        return (
          <rect
            key={i}
            x={i * barWidth}
            y={HEIGHT - barHeight}
            width={Math.max(1, barWidth - 1)}
            height={barHeight}
            fill={color}
          />
        );
      })}
    </svg>
  );
}
