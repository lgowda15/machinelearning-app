import { useEffect, useMemo, useState } from "react";
import { DistributionChart } from "../DistributionChart";
import { ImbalanceBanner } from "../ImbalanceBanner";
import { ScreenPanel } from "../ScreenPanel";
import { Sparkline } from "../Sparkline";
import { targetColorVarForDataType } from "../../lib/modelType";
import type { DataProfileResponse } from "../../hooks/useDataset";
import type { components } from "../../types/api";

type ColumnSummary = components["schemas"]["ColumnSummary"];

interface EdaScreenProps {
  profile: DataProfileResponse;
}

/**
 * Screen 2 -- Kaggle-style column explorer, no scroll (CLAUDE.md's layout
 * rule):
 * - top strip: rows/columns/target/data type, imbalance note if present
 * - left rail (~280px): scrollable column list -- name, type badge,
 *   missing %, sparkline. This is the screen's one scroll region, so a
 *   100-column dataset works unchanged.
 * - main panel: the selected column at full size, large distribution
 *   chart plus its stats.
 * Default selection is the target column; arrow keys move the selection.
 */
export function EdaScreen({ profile }: EdaScreenProps) {
  const [imbalanceVisible, setImbalanceVisible] = useState(true);

  const orderedColumns = useMemo(() => {
    const target = profile.columns.filter((c) => c.is_target);
    const rest = profile.columns.filter((c) => !c.is_target);

    return [...target, ...rest];
  }, [profile.columns]);

  const [selectedIndex, setSelectedIndex] = useState(0);

  // The target column gets the colour associated with the detected
  // data type. Other distributions remain in the default colour.
  const targetAccentVar = targetColorVarForDataType(profile.data_type);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIndex((i) => Math.min(orderedColumns.length - 1, i + 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIndex((i) => Math.max(0, i - 1));
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [orderedColumns.length]);

  const selected = orderedColumns[selectedIndex] ?? orderedColumns[0] ?? null;

  return (
    <ScreenPanel>
      {/* Top strip */}
      <div className="mb-3 flex shrink-0 flex-wrap items-baseline gap-x-4 gap-y-1 border-b border-rule pb-3">
        <h1 className="text-base font-semibold text-ink">EDA</h1>
        <TopStat label="Rows" value={profile.n_rows.toLocaleString()} />
        <TopStat label="Columns" value={profile.n_columns.toString()} />
        <TopStat label="Target" value={profile.target_column ?? "none"} />
        <TopStat label="Type" value={profile.data_type} />
      </div>

      {profile.class_imbalance && imbalanceVisible && (
        <ImbalanceBanner
          info={profile.class_imbalance}
          onDismiss={() => setImbalanceVisible(false)}
        />
      )}

      <div className="grid min-h-0 flex-1 grid-cols-[280px_1fr] gap-4">
        {/* Left rail -- the one scroll region */}
        <div
          role="listbox"
          aria-label="Columns"
          className="flex min-h-0 flex-col gap-1.5 overflow-y-auto rounded-panel border border-rule bg-ground p-2"
        >
          {orderedColumns.map((column, index) => (
            <button
              key={column.name}
              type="button"
              role="option"
              aria-selected={index === selectedIndex}
              onClick={() => setSelectedIndex(index)}
              className={`
                flex
                shrink-0
                cursor-pointer
                items-center
                justify-between
                gap-2
                rounded-panel
                border
                bg-surface
                px-2.5
                py-2
                text-left
                transition-colors
                ${index === selectedIndex ? "border-signal" : "border-transparent hover:border-rule"}
              `}
            >
              <div className="min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="truncate text-xs font-medium text-ink">{column.name}</span>
                  {column.is_target && (
                    <span
                      className="shrink-0 font-mono text-[9px] uppercase tracking-wide"
                      style={{ color: targetAccentVar }}
                    >
                      target
                    </span>
                  )}
                </div>
                <div className="mt-0.5 flex items-center gap-1.5 font-mono text-[9px] uppercase tracking-wide text-muted">
                  <span>{column.dtype}</span>
                  {column.missing_count > 0 && <span>· {(column.missing_pct * 100).toFixed(0)}% missing</span>}
                </div>
              </div>

              <Sparkline
                column={column}
                color={column.is_target ? targetAccentVar : "var(--color-muted)"}
              />
            </button>
          ))}
        </div>

        {/* Main panel -- selected column, full size. Never scrolls: the
            rail is this screen's one scroll region. */}
        <div className="min-h-0 overflow-hidden rounded-panel border border-rule bg-surface p-4">
          {selected && <ColumnDetail column={selected} accentVar={targetAccentVar} />}
        </div>
      </div>
    </ScreenPanel>
  );
}

/* -------------------------------------------------------------------------- */
/* Top strip stat                                                             */
/* -------------------------------------------------------------------------- */

function TopStat({ label, value }: { label: string; value: string }) {
  return (
    <span className="font-mono text-xs text-muted">
      {label} <span className="text-ink">{value}</span>
    </span>
  );
}

/* -------------------------------------------------------------------------- */
/* Selected column detail                                                     */
/* -------------------------------------------------------------------------- */

function ColumnDetail({
  column,
  accentVar,
}: {
  column: ColumnSummary;
  accentVar: string;
}) {
  return (
    <div>
      <DistributionChart column={column} targetAccentVar={accentVar} chartHeightClassName="h-36" />

      <dl className="mt-2 grid grid-cols-2 gap-1.5 sm:grid-cols-4">
        {column.dtype === "numeric" ? (
          <>
            <Stat label="Mean" value={column.mean != null ? column.mean.toFixed(3) : "—"} />
            <Stat label="Std dev" value={column.std != null ? column.std.toFixed(3) : "—"} />
            <Stat label="Min" value={column.min != null ? column.min.toFixed(3) : "—"} />
            <Stat label="Max" value={column.max != null ? column.max.toFixed(3) : "—"} />
          </>
        ) : (
          <Stat
            label="Most common"
            value={column.top_value != null ? `${column.top_value} (${column.top_value_freq})` : "—"}
          />
        )}
        <Stat label="Unique" value={column.unique_count.toString()} />
        <Stat label="Missing" value={`${(column.missing_pct * 100).toFixed(1)}% (${column.missing_count})`} />
      </dl>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-panel border border-rule bg-ground px-2.5 py-1.5">
      <p className="font-mono text-[10px] uppercase tracking-wide text-muted">{label}</p>
      <p className="truncate font-mono text-sm text-ink">{value}</p>
    </div>
  );
}
