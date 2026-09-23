import { typeBorderClass, typeTextClass } from "../lib/modelType";
import type { components } from "../types/api";

type ModelSummary = components["schemas"]["ModelSummary"];

interface ModelCardProps {
  model: ModelSummary;
  selected: boolean;
  focused: boolean;
  onSelect: () => void;
}

/**
 * Model Selection's compact card -- name + type badge only (screen 3's
 * master-detail layout, frontend.md). Default parameters live in the
 * detail panel for whichever card is focused, not on the card itself.
 * Clicking a card both toggles its selection and focuses it.
 */
export function ModelCard({
  model,
  selected,
  focused,
  onSelect,
}: ModelCardProps) {
  const typeText = typeTextClass(model.model_type);

  return (
    <button
      type="button"
      aria-pressed={selected}
      onClick={onSelect}
      className={`
        group
        flex
        w-full
        items-center
        justify-between
        gap-3
        rounded-panel
        border
        bg-surface
        px-4
        py-3
        text-left
        transition-all
        duration-150
        cursor-pointer
        ${
          selected
            ? `${typeBorderClass(model.model_type)} shadow-sm`
            : focused
              ? "border-ink/40"
              : "border-rule hover:border-ink/40"
        }
      `}
    >
      <div className="min-w-0">
        <h3 className="truncate text-sm font-medium text-ink">
          {model.model_name}
        </h3>
      </div>

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
          bg-ground
          ${typeText}
        `}
      >
        {model.model_type}
      </span>
    </button>
  );
}
