const MIN_TEST_PCT = 10;
const MAX_TEST_PCT = 50;
const STEP_PCT = 5;

interface SplitSliderProps {
  testSize: number; // fraction, e.g. 0.2 -- TrainRequest.test_size (backend.md)
  onChange: (testSize: number) => void;
  disabled?: boolean;
}

/** Train/test split control -- shared by Upload (screen 1) and its
 * confirmation on Training (screen 4), frontend.md's layout contract. */
export function SplitSlider({ testSize, onChange, disabled = false }: SplitSliderProps) {
  const testPct = Math.round(testSize * 100);
  const trainPct = 100 - testPct;

  return (
    <div>
      <div className="flex items-center justify-between text-sm">
        <span className="text-muted">Train/test split</span>
        <span className="font-mono text-ink">
          {trainPct}% / {testPct}%
        </span>
      </div>
      <input
        type="range"
        min={MIN_TEST_PCT}
        max={MAX_TEST_PCT}
        step={STEP_PCT}
        value={testPct}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value) / 100)}
        className="mt-2 w-full accent-signal disabled:opacity-50"
        aria-label="Test set percentage"
      />
    </div>
  );
}
