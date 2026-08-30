import { ModelCard } from "../ModelCard";
import { ScreenPanel, WORKSPACE_WIDTH } from "../ScreenPanel";
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

/** Screen 3 (frontend.md): card grid, 3 across, every model always visible.
 * Compatibility is a filter, not a recommendation (CLAUDE.md) -- incompatible
 * cards stay in place, greyed out, each with its reason. */
export function ModelSelectionScreen({
  models,
  compatibility,
  loading,
  error,
  selected,
  onToggle,
}: ModelSelectionScreenProps) {
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

  const incompatibleByKey = new Map(compatibility.incompatible.map((m) => [m.key, m.reason]));

  return (
    <ScreenPanel maxWidthClassName={WORKSPACE_WIDTH}>
      <h1 className="mb-1 text-lg font-medium text-ink">Model selection</h1>
      <p className="mb-4 text-sm text-muted">
        Detected data type: <span className="font-mono text-ink">{compatibility.dataType}</span>.
        Select one or more models to train.
      </p>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {models.map((model) => {
          const reason = incompatibleByKey.get(model.key);
          return (
            <ModelCard
              key={model.key}
              model={model}
              compatible={reason === undefined}
              reason={reason}
              selected={selected.includes(model.key)}
              onToggle={() => onToggle(model.key)}
            />
          );
        })}
      </div>
    </ScreenPanel>
  );
}
