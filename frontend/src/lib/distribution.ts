import type { components } from "../types/api";

type ColumnSummary = components["schemas"]["ColumnSummary"];

export interface DistributionBucket {
  label: string;
  count: number;
}

/**
 * Shapes a column's `distribution` (backend histogram/top-N payload,
 * CODING_STANDARDS.md SS7) into chart-ready buckets. Pure and separately
 * tested so the chart component itself stays a thin renderer.
 */
export function bucketsForColumn(column: ColumnSummary): DistributionBucket[] {
  const dist = column.distribution;
  if (dist.kind === "numeric") {
    return dist.counts.map((count, i) => ({
      label: `${dist.bin_edges[i].toFixed(1)}–${dist.bin_edges[i + 1].toFixed(1)}`,
      count,
    }));
  }
  const buckets = dist.categories.map((category, i) => ({
    label: category,
    count: dist.counts[i],
  }));
  if (dist.other_count > 0) {
    buckets.push({ label: "Other", count: dist.other_count });
  }
  return buckets;
}

/** A one-line text alternative for the chart's key number (quality floor, frontend.md). */
export function topBucketCaption(buckets: DistributionBucket[]): string | null {
  if (buckets.length === 0) return null;
  const top = buckets.reduce((a, b) => (b.count > a.count ? b : a));
  return `Most common: ${top.label} (n=${top.count})`;
}
