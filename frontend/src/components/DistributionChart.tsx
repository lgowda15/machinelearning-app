import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { bucketsForColumn, topBucketCaption } from "../lib/distribution";
import type { components } from "../types/api";

type ColumnSummary = components["schemas"]["ColumnSummary"];

const AXIS_TICK = { fontSize: 10, fontFamily: "var(--font-mono)", fill: "var(--color-muted)" };

interface DistributionChartProps {
  column: ColumnSummary;
  // The colour for the target column's bars/label -- the model type its
  // data_type will train (EdaScreen.tsx), not always --signal. Every
  // non-target column stays --ink regardless.
  targetAccentVar: string;
}

/** One column's distribution -- screen 2's right column, target first
 * (frontend.md). The target's chart is the "series in focus": it gets the
 * accent colour, everything else --ink. A text caption carries the same key
 * number for the quality floor's "text alternatives" requirement. */
export function DistributionChart({ column, targetAccentVar }: DistributionChartProps) {
  const buckets = bucketsForColumn(column);
  const caption = topBucketCaption(buckets);
  const barColor = column.is_target ? targetAccentVar : "var(--color-ink)";

  return (
    <div className="rounded-panel border border-rule bg-surface p-3">
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="text-sm font-medium text-ink">
          {column.name}
          {column.is_target && (
            <span className="ml-2 font-mono text-xs uppercase" style={{ color: targetAccentVar }}>
              target
            </span>
          )}
        </h3>
        {column.missing_count > 0 && (
          <span className="shrink-0 font-mono text-xs text-muted">
            {(column.missing_pct * 100).toFixed(0)}% missing
          </span>
        )}
      </div>
      {buckets.length > 0 ? (
        <div className="mt-2 h-40">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={buckets} margin={{ top: 4, right: 4, left: 4, bottom: 4 }}>
              <CartesianGrid stroke="var(--color-rule)" vertical={false} />
              <XAxis
                dataKey="label"
                tick={AXIS_TICK}
                interval={0}
                angle={-30}
                textAnchor="end"
                height={40}
              />
              <YAxis tick={AXIS_TICK} allowDecimals={false} width={32} />
              <Tooltip
                contentStyle={{
                  background: "var(--color-surface)",
                  border: "1px solid var(--color-rule)",
                  borderRadius: 4,
                  boxShadow: "none",
                  fontFamily: "var(--font-mono)",
                  fontSize: 12,
                }}
                cursor={{ fill: "var(--color-rule)", opacity: 0.3 }}
              />
              <Bar dataKey="count" fill={barColor} radius={0} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <p className="mt-2 text-sm text-muted">No values to chart.</p>
      )}
      {caption && <p className="mt-1 font-mono text-xs text-muted">{caption}</p>}
    </div>
  );
}
