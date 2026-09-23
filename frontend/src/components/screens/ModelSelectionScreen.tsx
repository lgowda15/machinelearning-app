import { useState } from "react";
import { ModelCard } from "../ModelCard";
import { ScreenHeader } from "../ScreenHeader";
import { ScreenPanel } from "../ScreenPanel";
import { typeBorderClass, typeTextClass } from "../../lib/modelType";
import type { Compatibility } from "../../hooks/useModels";
import type { components } from "../../types/api";

type ModelSummary = components["schemas"]["ModelSummary"];

interface ModelSelectionScreenProps {
  models: ModelSummary[] | null;
  compatibility: Compatibility | null;
  loading: boolean;
  error: Error | null;
  selected: string[];
  onToggle: (key: string) => void;
}

/**
 * Screen 3 -- master-detail, no scroll (CLAUDE.md's layout rule):
 * compatible models as compact cards on the left (colour-coded by type),
 * with incompatible models collapsed into one expandable strip beneath
 * them; a model's default parameters live in the right-hand detail panel
 * for whichever card is focused, not on the card itself. Compatibility is
 * a filter, not a recommendation (CLAUDE.md) -- incompatible models stay
 * visible, collapsed, each with its reason available on expand.
 */
export function ModelSelectionScreen({
  models,
  compatibility,
  loading,
  error,
  selected,
  onToggle,
}: ModelSelectionScreenProps) {
  const [focusedKey, setFocusedKey] = useState<string | null>(null);
  const [incompatibleExpanded, setIncompatibleExpanded] = useState(false);

  if (loading || !models) {
    return (
      <ScreenPanel>
        <p className="text-sm text-muted">Loading model registry…</p>
      </ScreenPanel>
    );
  }
  if (error) {
    return (
      <ScreenPanel>
        <p className="text-sm text-ink">Model registry request failed: {error.message}</p>
      </ScreenPanel>
    );
  }
  if (!compatibility) {
    return (
      <ScreenPanel>
        <p className="text-sm text-muted">Checking compatibility with this dataset…</p>
      </ScreenPanel>
    );
  }

  const { compatible, incompatible, dataType } = compatibility;

  const focusedModel =
    compatible.find((m) => m.key === focusedKey) ?? compatible[0] ?? null;

  return (
    <ScreenPanel>
      <ScreenHeader
        title="Model selection"
        description={`Detected data type: ${dataType}. Select one or more models to train.`}
      />

      <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)] gap-6">
        {/* Left: compact card grid + collapsed incompatible strip */}
        <div className="flex min-h-0 flex-col gap-3">
          <div className="grid min-h-0 flex-1 auto-rows-min gap-2.5 overflow-y-auto pr-1 sm:grid-cols-2">
            {compatible.map((model) => (
              <ModelCard
                key={model.key}
                model={model}
                selected={selected.includes(model.key)}
                focused={focusedModel?.key === model.key}
                onSelect={() => {
                  setFocusedKey(model.key);
                  onToggle(model.key);
                }}
              />
            ))}
          </div>

          {incompatible.length > 0 && (
            <div className="shrink-0 rounded-panel border border-rule bg-ground">
              <button
                type="button"
                onClick={() => setIncompatibleExpanded((v) => !v)}
                aria-expanded={incompatibleExpanded}
                className="flex w-full cursor-pointer items-center justify-between gap-3 px-4 py-3 text-left"
              >
                <span className="text-xs text-muted">
                  Not compatible with {dataType} data ({incompatible.length})
                </span>
                <span className="shrink-0 font-mono text-xs text-muted">
                  {incompatibleExpanded ? "−" : "+"}
                </span>
              </button>

              {incompatibleExpanded && (
                <div className="flex flex-wrap gap-2 border-t border-rule px-4 py-3">
                  {incompatible.map((model) => (
                    <span
                      key={model.key}
                      title={model.reason}
                      className="rounded-full border border-rule bg-surface px-3 py-1.5 text-xs text-muted"
                    >
                      <span className="font-medium text-ink">{model.model_name}</span>
                      {" — "}
                      {model.reason}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right: detail panel for the focused model */}
        <div className="min-h-0 overflow-y-auto rounded-panel border border-rule bg-ground p-5">
          {focusedModel ? (
            <ModelDetail
              model={focusedModel}
              selected={selected.includes(focusedModel.key)}
            />
          ) : (
            <p className="text-sm text-muted">No compatible models for this dataset.</p>
          )}
        </div>
      </div>
    </ScreenPanel>
  );
}

/* -------------------------------------------------------------------------- */
/* Detail panel                                                               */
/* -------------------------------------------------------------------------- */

function ModelDetail({
  model,
  selected,
}: {
  model: ModelSummary;
  selected: boolean;
}) {
  const hyperparamEntries = Object.entries(model.default_hyperparameters);
  const typeText = typeTextClass(model.model_type);

  return (
    <div>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="truncate text-base font-semibold text-ink">{model.model_name}</h2>
          <span className={`mt-1 inline-block font-mono text-xs uppercase tracking-wide ${typeText}`}>
            {model.model_type}
          </span>
        </div>

        <span
          className={`
            shrink-0
            rounded-panel
            border
            px-3
            py-1
            text-xs
            font-medium
            ${selected ? `${typeBorderClass(model.model_type)} ${typeText}` : "border-rule text-muted"}
          `}
        >
          {selected ? "✓ Selected" : "Available"}
        </span>
      </div>

      {hyperparamEntries.length > 0 ? (
        <div className="mt-5 border-t border-rule pt-4">
          <p className="mb-2 text-[10px] font-medium uppercase tracking-wider text-muted">
            Default parameters
          </p>

          <dl className="grid grid-cols-1 gap-x-6 gap-y-1.5 font-mono text-xs">
            {hyperparamEntries.map(([key, value]) => (
              <div key={key} className="flex min-w-0 justify-between gap-2">
                <dt className="truncate text-muted">{key}</dt>
                <dd className="truncate text-ink">{String(value)}</dd>
              </div>
            ))}
          </dl>
        </div>
      ) : (
        <p className="mt-5 border-t border-rule pt-4 text-xs text-muted">No configurable parameters.</p>
      )}
    </div>
  );
}
