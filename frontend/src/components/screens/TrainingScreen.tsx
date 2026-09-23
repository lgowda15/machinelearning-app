import { ScreenPanel } from "../ScreenPanel";
import { SplitSlider } from "../SplitSlider";
import { typeBorderClass, typeTextClass } from "../../lib/modelType";
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

/**
 * Screen 4:
 * - Confirm the train/test split
 * - Review selected models
 * - Run synchronous training
 * - Show pending/training/completed states
 */
export function TrainingScreen({
  dataId,
  models,
  selectedModelKeys,
  testSize,
  onTestSizeChange,
  trainingState,
}: TrainingScreenProps) {
  const { results, loading, error, train } = trainingState;

  const selectedModels = (models ?? []).filter((model) =>
    selectedModelKeys.includes(model.key),
  );

  if (selectedModels.length === 0) {
    return (
      <ScreenPanel maxWidthClassName="max-w-[1100px]">
        <div className="flex min-h-[420px] flex-col items-center justify-center text-center">
          <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-ground text-muted">
            —
          </div>

          <h1 className="text-base font-semibold text-ink">
            No models selected
          </h1>

          <p className="mt-2 max-w-md text-sm text-muted">
            Select at least one compatible model before starting training.
          </p>
        </div>
      </ScreenPanel>
    );
  }

  const handleTrain = () => {
    train({
      dataId,
      models: selectedModelKeys.map((key) => ({
        model_key: key,
        hyperparameters: {},
      })),
      testSize,
    });
  };

  const trainingComplete = !!results;

  return (
    <ScreenPanel maxWidthClassName="max-w-[1200px]">
      <div className="flex flex-col gap-4">
        {/* Header */}
        <div>
          <p className="font-mono text-[10px] uppercase tracking-wider text-muted">
            Step 4 · Model training
          </p>

          <h1 className="mt-1 text-xl font-semibold tracking-tight text-ink">
            Training
          </h1>

          <p className="mt-1 text-xs leading-relaxed text-muted">
            Review your configuration and train the selected models.
          </p>
        </div>

        {/* Main two-column workspace */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[0.85fr_1.15fr]">
          {/* LEFT — Training Configuration */}
          <section className="rounded-panel border border-rule bg-ground p-4">
            <div className="mb-3">
              <h2 className="text-sm font-semibold text-ink">
                Training configuration
              </h2>

              <p className="mt-1 text-[11px] text-muted">
                Choose how much data should be reserved for evaluation.
              </p>
            </div>

            <div className="rounded-panel border border-rule bg-surface p-3">
              <SplitSlider
                testSize={testSize}
                onChange={onTestSizeChange}
                disabled={loading}
              />
            </div>

            <div className="mt-3 grid grid-cols-2 gap-2">
              <div className="rounded-panel border border-rule bg-surface px-3 py-2.5">
                <p className="font-mono text-[9px] uppercase tracking-wide text-muted">
                  Training
                </p>
                <p className="mt-1 font-mono text-sm font-medium text-ink">
                  {(1 - testSize) * 100}%
                </p>
              </div>

              <div className="rounded-panel border border-rule bg-surface px-3 py-2.5">
                <p className="font-mono text-[9px] uppercase tracking-wide text-muted">
                  Testing
                </p>
                <p className="mt-1 font-mono text-sm font-medium text-ink">
                  {testSize * 100}%
                </p>
              </div>
            </div>
          </section>

          {/* RIGHT — Selected Models */}
          <section className="rounded-panel border border-rule bg-ground p-4">
            <div className="mb-3 flex items-center justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold text-ink">
                  Selected models
                </h2>

                <p className="mt-1 text-[11px] text-muted">
                  These models will be trained using the configuration above.
                </p>
              </div>

              <span className="shrink-0 rounded-full bg-surface px-3 py-1 font-mono text-xs text-muted">
                {selectedModels.length}
              </span>
            </div>

            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {selectedModels.map((model, index) => {
                const result = results?.results.find(
                  (r) => r.model_key === model.key,
                );

                const status = loading
                  ? "training"
                  : result
                    ? "done"
                    : "pending";

                return (
                  <ModelTrainingCard
                    key={model.key}
                    model={model}
                    index={index}
                    status={status}
                  />
                );
              })}
            </div>
          </section>
        </div>

        {/* Training Action */}
        <section
          className={`
            rounded-panel border p-4
            ${
              trainingComplete
                ? "border-rule bg-ground"
                : loading
                  ? "border-signal/40 bg-ground"
                  : "border-rule bg-surface"
            }
          `}
        >
          {!loading && !trainingComplete && (
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-sm font-semibold text-ink">
                  Ready to train
                </h2>

                <p className="mt-1 text-xs text-muted">
                  {selectedModels.length} selected model
                  {selectedModels.length === 1 ? "" : "s"} will be trained.
                </p>
              </div>

              <button
                type="button"
                onClick={handleTrain}
                className="
                  shrink-0 cursor-pointer
                  rounded-panel
                  border border-signal
                  bg-signal
                  px-5 py-2.5
                  text-sm font-medium text-surface
                  transition-all duration-150
                  hover:-translate-y-0.5
                  hover:opacity-90
                  active:translate-y-0
                "
              >
                Train selected models
              </button>
            </div>
          )}

          {loading && (
            <div className="flex items-center justify-between gap-4">
              <div>
                <h2 className="text-sm font-semibold text-ink">
                  Training in progress
                </h2>

                <p className="mt-1 text-xs text-muted">
                  Please wait while the selected models are trained.
                </p>
              </div>

              <div className="flex shrink-0 items-center gap-2">
                <span className="h-2 w-2 animate-pulse rounded-full bg-signal" />

                <span className="font-mono text-[10px] uppercase tracking-wide text-signal">
                  Training
                </span>
              </div>
            </div>
          )}

          {trainingComplete && !loading && (
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-sm font-semibold text-ink">
                  Training complete
                </h2>

                <p className="mt-1 text-xs text-muted">
                  All selected models finished successfully.
                </p>
              </div>

              <span className="shrink-0 rounded-full bg-ground px-3 py-1.5 font-mono text-[10px] uppercase tracking-wide text-signal">
                ✓ Complete
              </span>
            </div>
          )}
        </section>

        {/* Error */}
        {error && (
          <div className="rounded-panel border border-rule bg-ground px-4 py-3">
            <p className="text-xs font-medium text-ink">
              Training failed
            </p>

            <p className="mt-1 text-[11px] leading-relaxed text-muted">
              {error.message}
            </p>
          </div>
        )}

        {/* Continue hint */}
        {trainingComplete && !loading && !error && (
          <p className="text-center font-mono text-[10px] uppercase tracking-wide text-signal">
            Training complete · Continue to Results
          </p>
        )}
      </div>
    </ScreenPanel>
  );
}

/* -------------------------------------------------------------------------- */
/* Model training card                                                        */
/* -------------------------------------------------------------------------- */

function ModelTrainingCard({
  model,
  index,
  status,
}: {
  model: ModelSummary;
  index: number;
  status: "pending" | "training" | "done";
}) {
  const isDone = status === "done";
  const isTraining = status === "training";

  return (
    <div
      className={`
        rounded-panel
        border
        bg-surface
        p-3
        transition-all duration-150
        ${
          isDone
            ? typeBorderClass(model.model_type)
            : isTraining
              ? "border-signal shadow-sm"
              : "border-rule"
        }
      `}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2.5">
          {/* Number / status */}
          <div
            className={`
              flex h-7 w-7 shrink-0 items-center justify-center
              rounded-panel border
              font-mono text-[10px] font-medium
              ${
                isDone
                  ? `${typeBorderClass(model.model_type)} ${typeTextClass(model.model_type)}`
                  : isTraining
                    ? "border-signal text-signal"
                    : "border-rule text-muted"
              }
            `}
          >
            {isDone ? "✓" : index + 1}
          </div>

          <div className="min-w-0">
            <h3 className="truncate text-xs font-medium text-ink">
              {model.model_name}
            </h3>

            <p
              className={`
                mt-0.5
                font-mono
                text-[9px]
                uppercase
                tracking-wide
                ${isDone ? typeTextClass(model.model_type) : "text-muted"}
              `}
            >
              {model.model_type}
            </p>
          </div>
        </div>

        {/* Status */}
        <span
          className={`
            shrink-0
            font-mono
            text-[9px]
            uppercase
            tracking-wide
            ${
              isDone
                ? typeTextClass(model.model_type)
                : isTraining
                  ? "text-signal"
                  : "text-muted"
            }
          `}
        >
          {isTraining ? "Training…" : isDone ? "Complete" : "Pending"}
        </span>
      </div>
    </div>
  );
}