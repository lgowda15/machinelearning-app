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

/**
 * Model Selection card.
 * Every model remains visible. Compatible models can be selected;
 * incompatible models remain visible but disabled with a clear reason.
 */
export function ModelCard({
  model,
  compatible,
  reason,
  selected,
  onToggle,
}: ModelCardProps) {
  const hyperparamEntries = Object.entries(model.default_hyperparameters);
  const typeText = typeTextClass(model.model_type);

  return (
    <button
      type="button"
      disabled={!compatible}
      aria-pressed={selected}
      onClick={onToggle}
      className={`
        group
        flex
        w-full
        flex-col
        items-start
        rounded-panel
        border
        bg-surface
        p-5
        text-left
        transition-all
        duration-150
        ${
          !compatible
            ? "cursor-not-allowed border-rule opacity-55"
            : selected
              ? `${typeBorderClass(model.model_type)} shadow-sm`
              : "cursor-pointer border-rule hover:-translate-y-0.5 hover:border-ink/40 hover:shadow-sm"
        }
      `}
    >
      {/* Header */}
      <div className="flex w-full items-start justify-between gap-4">
        <div className="min-w-0">
          <h3 className="text-base font-semibold text-ink">
            {model.model_name}
          </h3>

          <p className="mt-1 text-xs text-muted">
            {compatible ? "Available for this dataset" : "Not compatible"}
          </p>
        </div>

        {/* Model type */}
        <span
          className={`
            shrink-0
            rounded-full
            px-2.5
            py-1
            font-mono
            text-[10px]
            font-medium
            uppercase
            tracking-wide
            ${
              compatible
                ? `${typeText} bg-ground`
                : "bg-ground text-muted"
            }
          `}
        >
          {model.model_type}
        </span>
      </div>

      {/* Hyperparameters */}
      {hyperparamEntries.length > 0 && (
        <div className="mt-4 w-full border-t border-rule pt-3">
          <p className="mb-2 text-[10px] font-medium uppercase tracking-wider text-muted">
            Default parameters
          </p>

          <dl className="grid grid-cols-1 gap-x-6 gap-y-1.5 font-mono text-xs sm:grid-cols-2">
            {hyperparamEntries.map(([key, value]) => (
              <div
                key={key}
                className="flex min-w-0 justify-between gap-2"
              >
                <dt className="truncate text-muted">{key}</dt>
                <dd className="truncate text-ink">{String(value)}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}

      {/* Incompatibility reason */}
      {!compatible && reason && (
        <div className="mt-4 w-full rounded-panel bg-ground px-3 py-2.5">
          <p className="text-xs leading-relaxed text-muted">
            {reason}
          </p>
        </div>
      )}

      {/* Selection state */}
      {compatible && (
        <div className="mt-4 flex w-full items-center justify-between border-t border-rule pt-3">
          <span
            className={`text-xs font-medium ${
              selected ? typeText : "text-muted"
            }`}
          >
            {selected ? "Selected" : "Available"}
          </span>

          <span
            className={`
              rounded-panel
              border
              px-3
              py-1
              text-xs
              font-medium
              transition-colors
              ${
                selected
                  ? `${typeBorderClass(model.model_type)} ${typeText}`
                  : "border-rule text-muted group-hover:border-ink/30 group-hover:text-ink"
              }
            `}
          >
            {selected ? "✓ Selected" : "Select"}
          </span>
        </div>
      )}
    </button>
  );
}