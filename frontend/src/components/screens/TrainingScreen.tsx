import { ScreenPanel } from "../ScreenPanel";
import { SplitSlider } from "../SplitSlider";
import type { useTraining } from "../../hooks/useTraining";
import type { components } from "../../types/api";

type ModelSummary = components["schemas"]["ModelSummary"];

interface TrainingScreenProps {
  dataId: string;
  models: ModelSummary[] | null;
  selectedModelKeys: string[];
  testSize: number;
  onTestSizeChange: (testSize: number) => void;
  trainingState: ReturnType<typeof useTraining>;
}

/** Screen 4 (frontend.md): selected models listed, split confirmation, one
 * primary action. Training is synchronous (ARCHITECTURE.md §6) -- the rows
 * below move from pending to done together when the single response lands,
 * never a polling UI. Full per-type results render on screen 5
 * (ResultsScreen); this screen's job ends at "done". */
export function TrainingScreen({
  dataId,
  models,
  selectedModelKeys,
  testSize,
  onTestSizeChange,
  trainingState,
}: TrainingScreenProps) {
  const { results, loading, error, train } = trainingState;
  const selectedModels = (models ?? []).filter((m) => selectedModelKeys.includes(m.key));

  if (selectedModels.length === 0) {
    return (
      <ScreenPanel>
        <p className="text-sm text-muted">Select at least one model to continue.</p>
      </ScreenPanel>
    );
  }

  const handleTrain = () => {
    train({
      dataId,
      models: selectedModelKeys.map((key) => ({ model_key: key, hyperparameters: {} })),
      testSize,
    });
  };

  return (
    <ScreenPanel>
      <div className="flex flex-col gap-4">
        <div>
          <h1 className="text-lg font-medium text-ink">Training</h1>
          <p className="mt-1 text-sm text-muted">
            {selectedModels.length} model{selectedModels.length === 1 ? "" : "s"} selected.
          </p>
        </div>

        <div className="rounded-panel border border-rule p-4">
          <SplitSlider testSize={testSize} onChange={onTestSizeChange} disabled={loading} />
        </div>

        <ul className="flex flex-col gap-2">
          {selectedModels.map((model) => {
            const result = results?.results.find((r) => r.model_key === model.key);
            const status = loading ? "training" : result ? "done" : "pending";
            return (
              <li
                key={model.key}
                className="flex items-center justify-between rounded-panel border border-rule px-4 py-2 text-sm"
              >
                <span className="text-ink">{model.model_name}</span>
                <span
                  className={
                    "font-mono text-xs uppercase " + (status === "done" ? "text-signal" : "text-muted")
                  }
                >
                  {status}
                </span>
              </li>
            );
          })}
        </ul>

        <button
          type="button"
          onClick={handleTrain}
          disabled={loading}
          className="rounded-panel border border-signal bg-signal px-4 py-2 text-sm font-medium text-surface disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Training…" : "Train selected models"}
        </button>

        {error && <p className="text-sm text-ink">Training failed: {error.message}</p>}

        {results && (
          <p className="font-mono text-xs text-signal">Training complete. Continue to Results.</p>
        )}
      </div>
    </ScreenPanel>
  );
}
