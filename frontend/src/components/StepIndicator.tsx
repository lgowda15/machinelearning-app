import { STEPS, type StepId } from "../types/steps";

interface StepIndicatorProps {
  // null on the Start screen -- nothing is "current" before step 1 begins.
  currentStep: StepId | null;
  // Index of the furthest step ever reached (App.tsx). A step at or before
  // this index has been completed and is navigable; -1 means nothing has
  // been completed yet.
  maxStepIndexReached: number;
  onNavigate: (index: number) => void;
}

/**
 * The shell header's step indicator (frontend.md's Navigation section):
 * clicking a completed step jumps directly there, clicking the current step
 * does nothing, and steps after the current one are shown but disabled --
 * later steps depend on choices (a selected model, a trained result) that
 * later ones haven't made yet.
 */
export function StepIndicator({ currentStep, maxStepIndexReached, onNavigate }: StepIndicatorProps) {
  return (
    <ol
      aria-label="Progress"
      className="flex min-w-0 flex-1 items-center gap-2 overflow-x-auto font-mono text-xs tracking-wide uppercase"
    >
      {STEPS.map((step, index) => {
        const isCurrent = step.id === currentStep;
        const isCompleted = !isCurrent && index <= maxStepIndexReached;

        return (
          <li key={step.id} className="flex shrink-0 items-center gap-2">
            <button
              type="button"
              aria-current={isCurrent ? "step" : undefined}
              disabled={!isCompleted}
              onClick={() => isCompleted && onNavigate(index)}
              className={
                "flex h-6 w-6 shrink-0 items-center justify-center rounded-panel border " +
                (isCurrent
                  ? "border-signal text-signal"
                  : isCompleted
                    ? "cursor-pointer border-white/40 text-white hover:border-signal hover:text-signal"
                    : "cursor-not-allowed border-white/15 text-white/35")
              }
            >
              {index + 1}
            </button>
            <span className={isCurrent ? "text-white" : isCompleted ? "text-white/70" : "text-white/35"}>
              {step.label}
            </span>
            {index < STEPS.length - 1 && (
              <span aria-hidden="true" className="mx-1 h-px w-6 bg-white/15" />
            )}
          </li>
        );
      })}
    </ol>
  );
}
