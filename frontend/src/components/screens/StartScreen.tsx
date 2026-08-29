import { Logo } from "../Logo";
import { ScreenPanel } from "../ScreenPanel";

interface StartScreenProps {
  onBegin: () => void;
}

/** The Start screen (frontend.md): full logo + banner, one short paragraph
 * naming what the tool does, single "Begin" action into Upload (step 1).
 * Precedes the numbered 1-7 flow -- orientation, not a dashboard. */
export function StartScreen({ onBegin }: StartScreenProps) {
  return (
    <ScreenPanel>
      <div className="flex flex-col items-center gap-6 py-4 text-center">
        <Logo variant="full" className="h-56 w-auto" />

        <div>
          <h1 className="text-lg font-medium text-ink">ML Integration Platform</h1>
          <p className="mt-2 text-sm text-muted">
            Upload a dataset, choose from the models registered on this platform, train them on a
            split you control, and compare what they produce. Every model is shown; the platform
            filters by what your data can run, never by which model to pick.
          </p>
        </div>

        <button
          type="button"
          onClick={onBegin}
          className="rounded-panel border border-signal bg-signal px-6 py-2 text-sm font-medium text-surface"
        >
          Begin
        </button>
      </div>
    </ScreenPanel>
  );
}
