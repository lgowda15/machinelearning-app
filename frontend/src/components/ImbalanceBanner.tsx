import type { components } from "../types/api";

type ImbalanceInfo = components["schemas"]["ImbalanceInfo"];

interface ImbalanceBannerProps {
  info: ImbalanceInfo;
  onDismiss: () => void;
}

/** Full-width, dismissible, shown only on the EDA screen (ARCHITECTURE.md
 * SS8) -- informational, never a blocker, never shown again after dismissal. */
export function ImbalanceBanner({ info, onDismiss }: ImbalanceBannerProps) {
  if (!info.is_imbalanced || !info.message) return null;

  return (
    <div className="mb-4 flex items-center justify-between gap-4 rounded-panel border border-rule bg-surface px-4 py-3 text-sm text-ink">
      <span>{info.message}</span>
      <button
        type="button"
        onClick={onDismiss}
        aria-label="Dismiss"
        className="shrink-0 text-muted hover:text-ink"
      >
        ✕
      </button>
    </div>
  );
}
