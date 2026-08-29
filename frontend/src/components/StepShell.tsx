import type { ReactNode } from "react";
import { Logo } from "./Logo";
import { StepIndicator } from "./StepIndicator";
import type { StepId, View } from "../types/steps";

interface StepShellProps {
  view: View;
  maxStepIndexReached: number;
  onNavigateToStep: (index: number) => void;
  canGoBack: boolean;
  canGoForward: boolean;
  onBack: () => void;
  onForward: () => void;
  renderStart: () => ReactNode;
  renderStep: (step: StepId) => ReactNode;
}

/**
 * The persistent shell (frontend.md's Layout section): a full-width
 * `--brand-navy` header with the clickable step indicator and the small
 * corner logo, present on every screen including Start; a `--ground`
 * content area below it that scrolls as one document, never a second
 * independent scroll container. Back/forward stay as the linear fallback --
 * the header indicator is the addition, not a replacement.
 */
export function StepShell({
  view,
  maxStepIndexReached,
  onNavigateToStep,
  canGoBack,
  canGoForward,
  onBack,
  onForward,
  renderStart,
  renderStep,
}: StepShellProps) {
  const isStart = view === "start";

  return (
    <div className="flex min-h-screen flex-col bg-ground">
      <header className="sticky top-0 z-20 flex h-14 shrink-0 items-center gap-4 border-b border-black/20 bg-brand-navy px-4">
        <StepIndicator
          currentStep={isStart ? null : view}
          maxStepIndexReached={isStart ? -1 : maxStepIndexReached}
          onNavigate={onNavigateToStep}
        />
        <Logo variant="mark" className="h-9 w-9" />
      </header>

      <main className="flex-1 px-4 py-8">{isStart ? renderStart() : renderStep(view)}</main>

      {!isStart && (
        <nav className="flex justify-between border-t border-rule bg-surface px-4 py-3">
          <button
            type="button"
            onClick={onBack}
            disabled={!canGoBack}
            className="rounded-panel border border-rule px-4 py-2 text-sm text-ink disabled:cursor-not-allowed disabled:text-muted disabled:opacity-50"
          >
            Back
          </button>
          <button
            type="button"
            onClick={onForward}
            disabled={!canGoForward}
            className="rounded-panel border border-signal px-4 py-2 text-sm text-signal disabled:cursor-not-allowed disabled:border-rule disabled:text-muted disabled:opacity-50"
          >
            Continue
          </button>
        </nav>
      )}
    </div>
  );
}
