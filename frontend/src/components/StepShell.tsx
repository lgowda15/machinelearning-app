import type { ReactNode } from "react";
import { DownloadReportButton } from "./DownloadReportButton";
import { Logo } from "./Logo";
import { StepIndicator } from "./StepIndicator";
import type { CycleState } from "../lib/report/types";
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
  cycle: CycleState;
}

/**
 * Persistent application shell -- a fixed three-row grid (header / main /
 * footer) filling the viewport. Only `main` can ever scroll, and it doesn't
 * on its own (`overflow-hidden`): each screen owns at most one internal
 * scroll region itself (CLAUDE.md's layout rule), so the page as a whole
 * never scrolls.
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
  cycle,
}: StepShellProps) {
  const isStart = view === "start";

  return (
    <div className="grid h-dvh grid-rows-[auto_1fr_auto] bg-ground">
      {/* Header */}
      <header className="relative flex h-14 shrink-0 items-center border-b border-black/20 bg-navy-900 px-6">
  {/* Centered step navigation */}
  <div className="absolute left-1/2 w-max max-w-[calc(100%-7rem)] -translate-x-1/2">
    <StepIndicator
      currentStep={isStart ? null : view}
      maxStepIndexReached={isStart ? -1 : maxStepIndexReached}
      onNavigate={onNavigateToStep}
    />
  </div>

  {/* Download report and logo stay on the right */}
  <div className="ml-auto flex items-center gap-4">
    {!isStart && <DownloadReportButton cycle={cycle} />}
    <Logo variant="mark" className="h-9 w-9" />
  </div>
</header>

      {/* Main content -- fills the row between header and footer exactly;
          it does not scroll itself, each screen does (or doesn't). */}
      <main className="min-h-0 overflow-hidden px-6 py-4">
        {isStart ? renderStart() : renderStep(view)}
      </main>

      {/* Bottom navigation -- a real grid row now, not fixed/floating. */}
      {!isStart && (
        <nav
          className="
            border-t
            border-rule
            bg-surface
            px-6
            py-3
          "
        >
          <div className="flex items-center justify-between">
            {/* Back */}
            <button
              type="button"
              onClick={onBack}
              disabled={!canGoBack}
              className="
                rounded-panel
                border
                border-rule
                px-4
                py-2
                text-sm
                text-ink
                transition-all
                duration-150
                cursor-pointer
                hover:bg-ground
                hover:border-ink/30
                active:scale-[0.98]
                disabled:cursor-not-allowed
                disabled:text-muted
                disabled:opacity-50
                disabled:hover:bg-transparent
                disabled:hover:border-rule
              "
            >
              Back
            </button>

            {/* Continue */}
            <button
              type="button"
              onClick={onForward}
              disabled={!canGoForward}
              className={`
                rounded-panel
                border
                px-5
                py-2
                text-sm
                font-medium
                transition-all
                duration-150
                ${
                  canGoForward
                    ? `
                      cursor-pointer
                      border-orange-700
                      bg-orange-700
                      text-white
                      hover:opacity-90
                      active:scale-[0.98]
                    `
                    : `
                      cursor-not-allowed
                      border-rule
                      bg-ground
                      text-muted
                      opacity-60
                    `
                }
              `}
            >
              Continue
            </button>
          </div>
        </nav>
      )}
    </div>
  );
}