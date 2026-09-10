import { useState } from "react";
import { Dropzone } from "../Dropzone";
import { ScreenPanel } from "../ScreenPanel";
import {
  buildSingleRowCsv,
  diffColumns,
  parseCsvHeader,
  type ColumnMismatch,
} from "../../lib/columns";
import { usePrediction } from "../../hooks/usePrediction";
import type { DataProfileResponse } from "../../hooks/useDataset";
import type { components } from "../../types/api";

type TrainResponse = components["schemas"]["TrainResponse"];
type ColumnSummary = components["schemas"]["ColumnSummary"];

const MAX_ROWS_SHOWN = 25;
type PredictMode = "csv" | "manual";

interface PredictScreenProps {
  profile: DataProfileResponse;
  trainingResults: TrainResponse | null;
}

/**
 * Screen 6:
 * - Choose a trained model
 * - Choose between CSV upload and manual entry
 * - Enter feature values in a compact responsive grid
 * - Display prediction results clearly
 */
export function PredictScreen({
  profile,
  trainingResults,
}: PredictScreenProps) {
  const { result, loading, error, predict, reset } = usePrediction();

  const [modelKey, setModelKey] = useState(
    trainingResults?.results[0]?.model_key ?? "",
  );

  const [mode, setMode] = useState<PredictMode>("csv");

  const [file, setFile] = useState<File | null>(null);
  const [mismatch, setMismatch] = useState<ColumnMismatch | null>(null);

  const [manualValues, setManualValues] = useState<Record<string, string>>(
    {},
  );

  const [manualMissing, setManualMissing] = useState<string[]>([]);

  if (!trainingResults || trainingResults.results.length === 0) {
    return (
      <ScreenPanel>
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-ground text-muted">
            —
          </div>

          <h1 className="text-base font-semibold text-ink">
            No trained model available
          </h1>

          <p className="mt-2 max-w-md text-sm text-muted">
            Train at least one model before predicting on new data.
          </p>
        </div>
      </ScreenPanel>
    );
  }

  const featureColumns: ColumnSummary[] = profile.columns.filter(
    (column) => column.name !== profile.target_column,
  );

  const expectedColumns = featureColumns.map((column) => column.name);

  const selectedModel = trainingResults.results.find(
    (model) => model.model_key === modelKey,
  );

  const switchMode = (next: PredictMode) => {
    setMode(next);
    reset();
    setMismatch(null);
    setManualMissing([]);
  };

  const handleFile = async (nextFile: File) => {
    reset();
    setFile(nextFile);

    const header = parseCsvHeader(await nextFile.text());
    setMismatch(diffColumns(expectedColumns, header));
  };

  const handlePredictCsv = () => {
    if (!file || mismatch) return;

    predict(trainingResults.training_id, modelKey, file);
  };

  const handleManualFieldChange = (
    name: string,
    value: string,
  ) => {
    setManualValues((prev) => ({
      ...prev,
      [name]: value,
    }));

    setManualMissing((prev) =>
      prev.filter((n) => n !== name),
    );
  };

  const handlePredictManual = () => {
    const missing = featureColumns
      .filter(
        (column) =>
          !(manualValues[column.name] ?? "").trim(),
      )
      .map((column) => column.name);

    if (missing.length > 0) {
      setManualMissing(missing);
      return;
    }

    setManualMissing([]);

    const csv = buildSingleRowCsv(
      expectedColumns,
      manualValues,
    );

    const manualFile = new File(
      [csv],
      "manual-entry.csv",
      { type: "text/csv" },
    );

    predict(
      trainingResults.training_id,
      modelKey,
      manualFile,
    );
  };

  const canPredictCsv =
    !!file && !mismatch && !loading;

  return (
    <ScreenPanel>
      <div className="flex flex-col gap-6">
        {/* Header */}
        <div>
          <p className="font-mono text-xs uppercase tracking-wider text-muted">
            New prediction
          </p>

          <h1 className="mt-1 text-xl font-semibold text-ink">
            Predict on new data
          </h1>

          <p className="mt-1 max-w-2xl text-sm leading-relaxed text-muted">
            Run a trained model on new data. Upload a CSV with the
            same feature columns as your training data, or enter one
            row manually.
          </p>
        </div>

        {/* Model selection */}
        <section className="rounded-panel border border-rule bg-ground p-5">
          <div className="mb-3">
            <h2 className="text-sm font-semibold text-ink">
              Model
            </h2>

            <p className="mt-1 text-xs text-muted">
              Choose which trained model should generate the
              prediction.
            </p>
          </div>

          <select
            id="predict-model"
            className="
              w-full
              cursor-pointer
              rounded-panel
              border
              border-rule
              bg-surface
              px-3
              py-2.5
              text-sm
              text-ink
              transition-colors
              hover:border-ink/30
              focus:border-signal
            "
            value={modelKey}
            onChange={(e) => {
              setModelKey(e.target.value);
              reset();
            }}
          >
            {trainingResults.results.map((model) => (
              <option
                key={model.model_key}
                value={model.model_key}
              >
                {model.model_name} · {model.model_type}
              </option>
            ))}
          </select>

          {selectedModel && (
            <div className="mt-3 flex flex-wrap gap-2">
              <span className="rounded-full bg-surface px-2.5 py-1 font-mono text-[10px] uppercase tracking-wide text-muted">
                {selectedModel.model_type}
              </span>

              <span className="rounded-full bg-surface px-2.5 py-1 font-mono text-[10px] uppercase tracking-wide text-muted">
                {selectedModel.n_features} features
              </span>
            </div>
          )}
        </section>

        {/* Input mode */}
        <section>
          <div className="mb-3">
            <h2 className="text-sm font-semibold text-ink">
              Input data
            </h2>

            <p className="mt-1 text-xs text-muted">
              Choose how you want to provide the new data.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-2 rounded-panel border border-rule bg-ground p-1.5">
            <button
              type="button"
              aria-pressed={mode === "csv"}
              onClick={() => switchMode("csv")}
              className={`
                cursor-pointer
                rounded-panel
                px-4
                py-2.5
                text-sm
                font-medium
                transition-all
                duration-150
                ${
                  mode === "csv"
                    ? "bg-surface text-ink shadow-sm"
                    : "text-muted hover:text-ink"
                }
              `}
            >
              Upload CSV
            </button>

            <button
              type="button"
              aria-pressed={mode === "manual"}
              onClick={() => switchMode("manual")}
              className={`
                cursor-pointer
                rounded-panel
                px-4
                py-2.5
                text-sm
                font-medium
                transition-all
                duration-150
                ${
                  mode === "manual"
                    ? "bg-surface text-ink shadow-sm"
                    : "text-muted hover:text-ink"
                }
              `}
            >
              Enter values
            </button>
          </div>
        </section>

        {/* CSV mode */}
        {mode === "csv" ? (
          <section className="rounded-panel border border-rule bg-surface p-5">
            <div className="mb-4">
              <h2 className="text-sm font-semibold text-ink">
                Upload prediction data
              </h2>

              <p className="mt-1 text-xs leading-relaxed text-muted">
                Your CSV should contain the same feature columns
                used to train the selected model.
              </p>
            </div>

            <Dropzone
              onFile={handleFile}
              loading={loading}
              fileName={file?.name}
            />

            {/* Column mismatch */}
            {mismatch && (
              <div className="mt-4 rounded-panel border border-rule bg-ground px-4 py-3">
                <p className="text-sm font-medium text-ink">
                  Column mismatch
                </p>

                <p className="mt-1 text-xs leading-relaxed text-muted">
                  This file's columns don't match the training
                  data.
                </p>

                {mismatch.missing.length > 0 && (
                  <p className="mt-2 text-xs text-muted">
                    Missing:{" "}
                    <span className="font-mono text-ink">
                      {mismatch.missing.join(", ")}
                    </span>
                  </p>
                )}

                {mismatch.unexpected.length > 0 && (
                  <p className="mt-1 text-xs text-muted">
                    Unexpected:{" "}
                    <span className="font-mono text-ink">
                      {mismatch.unexpected.join(", ")}
                    </span>
                  </p>
                )}
              </div>
            )}

            <div className="mt-5 flex justify-end">
              <button
                type="button"
                onClick={handlePredictCsv}
                disabled={!canPredictCsv}
                className={`
                  rounded-panel
                  border
                  px-5
                  py-2.5
                  text-sm
                  font-medium
                  transition-all
                  duration-150
                  ${
                    canPredictCsv
                      ? "cursor-pointer border-signal bg-signal text-surface hover:opacity-90 active:scale-[0.98]"
                      : "cursor-not-allowed border-rule bg-ground text-muted opacity-60"
                  }
                `}
              >
                {loading ? "Predicting…" : "Predict"}
              </button>
            </div>
          </section>
        ) : (
          /* Manual mode */
          <section className="rounded-panel border border-rule bg-surface p-5">
            <div className="mb-5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-sm font-semibold text-ink">
                    Enter feature values
                  </h2>

                  <p className="mt-1 text-xs text-muted">
                    Provide one value for every feature.
                  </p>
                </div>

                <span className="shrink-0 rounded-full bg-ground px-2.5 py-1 font-mono text-[10px] uppercase tracking-wide text-muted">
                  {featureColumns.length} features
                </span>
              </div>
            </div>

            {/* Feature grid */}
            <div className="grid gap-4 sm:grid-cols-2">
              {featureColumns.map((column, index) => {
                const isMissing = manualMissing.includes(
                  column.name,
                );

                return (
                  <div
                    key={column.name}
                    className={`
                      rounded-panel
                      border
                      bg-ground
                      p-3.5
                      transition-colors
                      ${
                        isMissing
                          ? "border-signal"
                          : "border-rule"
                      }
                    `}
                  >
                    <div className="mb-2 flex items-start justify-between gap-2">
                      <label
                        htmlFor={`manual-${column.name}`}
                        className="min-w-0 text-xs font-medium text-ink"
                      >
                        <span className="mr-1 font-mono text-[10px] text-muted">
                          {String(index + 1).padStart(2, "0")}
                        </span>

                        {column.name}
                      </label>

                      <span className="shrink-0 font-mono text-[10px] lowercase text-muted">
                        {column.dtype}
                      </span>
                    </div>

                    <input
                      id={`manual-${column.name}`}
                      type={
                        column.dtype === "numeric"
                          ? "number"
                          : "text"
                      }
                      step={
                        column.dtype === "numeric"
                          ? "any"
                          : undefined
                      }
                      value={
                        manualValues[column.name] ?? ""
                      }
                      disabled={loading}
                      placeholder={
                        column.dtype === "numeric"
                          ? "Enter number"
                          : "Enter value"
                      }
                      onChange={(e) =>
                        handleManualFieldChange(
                          column.name,
                          e.target.value,
                        )
                      }
                      className={`
                        w-full
                        rounded-panel
                        border
                        bg-surface
                        px-3
                        py-2.5
                        font-mono
                        text-sm
                        text-ink
                        placeholder:text-muted/50
                        transition-colors
                        focus:outline-none
                        ${
                          isMissing
                            ? "border-signal"
                            : "border-rule focus:border-signal"
                        }
                      `}
                    />
                  </div>
                );
              })}
            </div>

            {/* Missing fields */}
            {manualMissing.length > 0 && (
              <div className="mt-5 rounded-panel border border-rule bg-ground px-4 py-3">
                <p className="text-sm font-medium text-ink">
                  Complete the required fields
                </p>

                <p className="mt-1 text-xs leading-relaxed text-muted">
                  Enter a value for every feature before
                  predicting.
                </p>

                <p className="mt-2 text-xs text-muted">
                  Missing:{" "}
                  <span className="font-mono text-ink">
                    {manualMissing.join(", ")}
                  </span>
                </p>
              </div>
            )}

            <div className="mt-5 flex justify-end">
              <button
                type="button"
                onClick={handlePredictManual}
                disabled={loading}
                className={`
                  rounded-panel
                  border
                  px-5
                  py-2.5
                  text-sm
                  font-medium
                  transition-all
                  duration-150
                  ${
                    !loading
                      ? "cursor-pointer border-signal bg-signal text-surface hover:opacity-90 active:scale-[0.98]"
                      : "cursor-not-allowed border-rule bg-ground text-muted opacity-60"
                  }
                `}
              >
                {loading ? "Predicting…" : "Predict"}
              </button>
            </div>
          </section>
        )}

        {/* Error */}
        {error && (
          <div className="rounded-panel border border-rule bg-ground px-4 py-3">
            <p className="text-sm font-medium text-ink">
              Prediction failed
            </p>

            <p className="mt-1 text-xs leading-relaxed text-muted">
              {error.message}
            </p>
          </div>
        )}

        {/* Results */}
        {result && <PredictionTable result={result} />}
      </div>
    </ScreenPanel>
  );
}

/* -------------------------------------------------------------------------- */
/* Prediction formatting                                                     */
/* -------------------------------------------------------------------------- */

function formatPrediction(value: unknown): string {
  if (Array.isArray(value)) {
    return value
      .map((v) =>
        typeof v === "number"
          ? v.toFixed(4)
          : String(v),
      )
      .join(", ");
  }

  if (typeof value === "number") {
    return Number.isInteger(value)
      ? String(value)
      : value.toFixed(4);
  }

  return String(value);
}

/* -------------------------------------------------------------------------- */
/* Prediction results                                                         */
/* -------------------------------------------------------------------------- */

function PredictionTable({
  result,
}: {
  result: components["schemas"]["PredictionResponse"];
}) {
  const shown = result.predictions.slice(
    0,
    MAX_ROWS_SHOWN,
  );

  return (
    <section className="rounded-panel border border-rule bg-surface p-5">
      {/* Result header */}
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="font-mono text-xs uppercase tracking-wider text-muted">
            Prediction complete
          </p>

          <h2 className="mt-1 text-base font-semibold text-ink">
            Prediction results
          </h2>
        </div>

        <span className="rounded-full bg-ground px-3 py-1.5 font-mono text-xs text-ink">
          {result.n_samples}{" "}
          {result.n_samples === 1
            ? "prediction"
            : "predictions"}
        </span>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-panel border border-rule">
        <table className="w-full min-w-[520px] text-left text-sm">
          <thead>
            <tr className="border-b border-rule bg-ground">
              <th className="px-4 py-3 text-xs font-medium uppercase tracking-wide text-muted">
                Row
              </th>

              <th className="px-4 py-3 text-xs font-medium uppercase tracking-wide text-muted">
                Prediction
              </th>

              {result.probabilities && (
                <th className="px-4 py-3 text-xs font-medium uppercase tracking-wide text-muted">
                  Probabilities
                </th>
              )}
            </tr>
          </thead>

          <tbody>
            {shown.map((prediction, i) => (
              <tr
                key={i}
                className="border-b border-rule last:border-b-0"
              >
                <td className="px-4 py-3 font-mono text-xs text-muted">
                  {String(i).padStart(2, "0")}
                </td>

                <td className="px-4 py-3 font-mono text-sm text-ink">
                  {formatPrediction(prediction)}
                </td>

                {result.probabilities && (
                  <td className="px-4 py-3 font-mono text-xs text-ink">
                    {result.probabilities[i]
                      .map((p) => p.toFixed(3))
                      .join(", ")}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {result.n_samples > MAX_ROWS_SHOWN && (
        <p className="mt-2 font-mono text-xs text-muted">
          Showing first {MAX_ROWS_SHOWN} of{" "}
          {result.n_samples}.
        </p>
      )}
    </section>
  );
}