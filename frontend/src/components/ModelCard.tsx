import { typeBorderClass, typeTextClass } from "../lib/modelType";
import type { components } from "../types/api";

type ModelSummary = components["schemas"]["ModelSummary"];

interface ModelCardProps {
  model: ModelSummary;
  compatible: boolean;
  reason?: string;
  selected: boolean;
  onToggle: () => void;
}

/** Screen 3's card (frontend.md). Every model always renders; an
 * incompatible one stays visible at reduced opacity with a --rule outline
 * and one line naming why -- never hidden, never ranked (CLAUDE.md's "no
 * AutoML, no recommendations"). */
export function ModelCard({ model, compatible, reason, selected, onToggle }: ModelCardProps) {
  const hyperparamEntries = Object.entries(model.default_hyperparameters);
  // The model-type colour coding (frontend.md): "its card border/accent in
  // Model Selection" -- selected cards get a full border in their type's
  // colour; the type badge always carries it, compatible or not, since the
  // type itself isn't what makes a model incompatible.
  const typeText = typeTextClass(model.model_type);

  return (
    <button
      type="button"
      disabled={!compatible}
      aria-pressed={selected}
      onClick={onToggle}
      className={
        "flex flex-col items-start gap-2 rounded-panel border bg-surface p-4 text-left " +
        (!compatible
          ? "cursor-not-allowed border-rule opacity-50"
          : selected
            ? typeBorderClass(model.model_type)
            : "border-rule hover:border-ink")
      }
    >
      <div className="flex w-full items-center justify-between gap-2">
        <h3 className="text-sm font-medium text-ink">{model.model_name}</h3>
        <span className={"shrink-0 font-mono text-xs uppercase " + (compatible ? typeText : "text-muted")}>
          {model.model_type}
        </span>
      </div>

      {hyperparamEntries.length > 0 && (
        <dl className="w-full font-mono text-xs text-muted">
          {hyperparamEntries.map(([key, value]) => (
            <div key={key} className="flex gap-1">
              <dt>{key}:</dt>
              <dd className="text-ink">{String(value)}</dd>
            </div>
          ))}
        </dl>
      )}

      {!compatible && reason && <p className="text-xs text-muted">{reason}</p>}

      {compatible && (
        <span className={"font-mono text-xs " + (selected ? typeText : "text-muted")}>
          {selected ? "Selected" : "Select"}
        </span>
      )}
    </button>
  );
}
