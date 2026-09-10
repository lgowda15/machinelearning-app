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
 * Persistent application shell:
 * - Sticky header
 * - Scrollable page content
 * - Fixed bottom navigation
 * - Back and Continue remain accessible while scrolling
 * - Continue is highlighted only when forward navigation is available
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
      {/* Header */}
      <header className="sticky top-0 z-20 flex h-14 shrink-0 items-center border-b border-black/20 bg-brand-navy px-6">
  {/* Centered step navigation */}
  <div className="absolute left-1/2 w-max max-w-[calc(100%-7rem)] -translate-x-1/2">
    <StepIndicator
      currentStep={isStart ? null : view}
      maxStepIndexReached={isStart ? -1 : maxStepIndexReached}
      onNavigate={onNavigateToStep}
    />
  </div>

  {/* Logo stays on the right */}
  <div className="ml-auto">
    <Logo variant="mark" className="h-9 w-9" />
  </div>
</header>

      {/* Main content */}
      <main className="flex-1 px-6 py-8 pb-28">
        {isStart ? renderStart() : renderStep(view)}
      </main>

      {/* Fixed bottom navigation */}
      {!isStart && (
        <nav
          className="
            fixed
            bottom-0
            left-0
            right-0
            z-30
            border-t
            border-rule
            bg-surface/95
            px-6
            py-3
            shadow-sm
            backdrop-blur
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
                      border-signal
                      bg-signal
                      text-surface
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