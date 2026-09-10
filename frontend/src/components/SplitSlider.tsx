const MIN_TEST_PCT = 10;
const MAX_TEST_PCT = 50;
const STEP_PCT = 5;

interface SplitSliderProps {
  testSize: number;
  onChange: (testSize: number) => void;
  disabled?: boolean;
}

/** Train/test split control -- shared by Upload (screen 1) and
 * its confirmation on Training (screen 4). */
export function SplitSlider({
  testSize,
  onChange,
  disabled = false,
}: SplitSliderProps) {
  const testPct = Math.round(testSize * 100);
  const trainPct = 100 - testPct;

  return (
    <div className="w-full">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-sm font-medium text-ink">Train / test split</p>
          <p className="mt-0.5 text-xs text-muted">
            Choose how much data to reserve for testing.
          </p>
        </div>

        <div className="shrink-0 rounded-panel border border-rule bg-ground px-3 py-1.5 font-mono text-xs text-ink">
          <span>{trainPct}%</span>
          <span className="mx-1.5 text-muted">/</span>
          <span>{testPct}%</span>
        </div>
      </div>

      <input
        type="range"
        min={MIN_TEST_PCT}
        max={MAX_TEST_PCT}
        step={STEP_PCT}
        value={testPct}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value) / 100)}
        className="mt-4 w-full cursor-pointer accent-signal disabled:cursor-not-allowed disabled:opacity-50"
        aria-label="Test set percentage"
      />

      <div className="mt-1 flex justify-between font-mono text-[10px] text-muted">
  <span>10% test</span>
  <span>50% test</span>
</div>
    </div>
  );
}