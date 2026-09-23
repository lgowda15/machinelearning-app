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
  const [manualValues, setManualValues] = useState<Record<string, string>>({});
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
      prev.filter((item) => item !== name),
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
      <div className="flex flex-col gap-4">

        {/* Header */}
        <div>
          <p className="font-mono text-[10px] uppercase tracking-wider text-muted">
            Step 6 · New prediction
          </p>

          <h1 className="mt-1 text-xl font-semibold tracking-tight text-ink">
            Predict on new data
          </h1>

          <p className="mt-1 text-xs leading-relaxed text-muted">
            Select a trained model and provide new data for prediction.
          </p>
        </div>

        {/* Model */}
        <section className="rounded-panel border border-rule bg-ground p-4">
          <div className="flex flex-wrap items-center gap-3">
            <div className="mr-auto">
              <h2 className="text-sm font-semibold text-ink">
                Model
              </h2>
              <p className="mt-0.5 text-[10px] text-muted">
                Choose a trained model.
              </p>
            </div>

            <select
              id="predict-model"
              value={modelKey}
              onChange={(e) => {
                setModelKey(e.target.value);
                reset();
              }}
              className="
                min-w-[220px]
                cursor-pointer
                rounded-panel
                border
                border-rule
                bg-surface
                px-3
                py-2
                text-sm
                text-ink
                hover:border-ink/30
                focus:border-signal
                focus:outline-none
              "
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
              <>
                <span className="rounded-full bg-surface px-2.5 py-1 font-mono text-[9px] uppercase text-muted">
                  {selectedModel.model_type}
                </span>

                <span className="rounded-full bg-surface px-2.5 py-1 font-mono text-[9px] uppercase text-muted">
                  {selectedModel.n_features} features
                </span>
              </>
            )}
          </div>
        </section>

        {/* Input */}
        <section>
          <div className="mb-2">
            <h2 className="text-sm font-semibold text-ink">
              Input data
            </h2>
            <p className="mt-0.5 text-[10px] text-muted">
              Upload a CSV or enter values manually.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-1 rounded-panel border border-rule bg-ground p-1">
            <button
              type="button"
              aria-pressed={mode === "csv"}
              onClick={() => switchMode("csv")}
              className={`
                cursor-pointer
                rounded-panel
                px-3
                py-2
                text-xs
                font-medium
                transition-all
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
                px-3
                py-2
                text-xs
                font-medium
                transition-all
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

        {/* CSV */}
        {mode === "csv" ? (
          <section className="rounded-panel border border-rule bg-surface p-4">
            <div className="mb-3">
              <h2 className="text-sm font-semibold text-ink">
                Upload prediction data
              </h2>
              <p className="mt-0.5 text-[10px] text-muted">
                CSV must contain the same feature columns used during training.
              </p>
            </div>

            <Dropzone
              onFile={handleFile}
              loading={loading}
              fileName={file?.name}
            />

            {mismatch && (
              <div className="mt-3 rounded-panel border border-rule bg-ground px-3 py-2.5">
                <p className="text-xs font-medium text-ink">
                  Column mismatch
                </p>

                {mismatch.missing.length > 0 && (
                  <p className="mt-1 text-[10px] text-muted">
                    Missing:{" "}
                    <span className="font-mono text-ink">
                      {mismatch.missing.join(", ")}
                    </span>
                  </p>
                )}

                {mismatch.unexpected.length > 0 && (
                  <p className="mt-1 text-[10px] text-muted">
                    Unexpected:{" "}
                    <span className="font-mono text-ink">
                      {mismatch.unexpected.join(", ")}
                    </span>
                  </p>
                )}
              </div>
            )}

            <div className="mt-3 flex justify-end">
              <button
                type="button"
                onClick={handlePredictCsv}
                disabled={!canPredictCsv}
                className={`
                  rounded-panel
                  border
                  px-4
                  py-2
                  text-xs
                  font-medium
                  transition-all
                  ${
                    canPredictCsv
                      ? "cursor-pointer border-signal bg-signal text-white hover:opacity-90"
                      : "cursor-not-allowed border-rule bg-ground text-muted opacity-60"
                  }
                `}
              >
                {loading ? "Predicting…" : "Predict →"}
              </button>
            </div>
          </section>
        ) : (
          /* Manual */
          <section className="rounded-panel border border-rule bg-surface p-4">
            <div className="mb-3 flex items-center justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold text-ink">
                  Enter feature values
                </h2>
                <p className="mt-0.5 text-[10px] text-muted">
                  Provide one value for every feature.
                </p>
              </div>

              <span className="rounded-full bg-ground px-2.5 py-1 font-mono text-[9px] uppercase text-muted">
                {featureColumns.length} features
              </span>
            </div>

            <div className="grid gap-2 sm:grid-cols-2">
              {featureColumns.map((column, index) => {
                const isMissing =
                  manualMissing.includes(column.name);

                return (
                  <div
                    key={column.name}
                    className={`
                      rounded-panel
                      border
                      bg-ground
                      p-3
                      ${
                        isMissing
                          ? "border-signal"
                          : "border-rule"
                      }
                    `}
                  >
                    <div className="mb-1.5 flex items-start justify-between gap-2">
                      <label
                        htmlFor={`manual-${column.name}`}
                        className="min-w-0 text-xs font-medium text-ink"
                      >
                        <span className="mr-1 font-mono text-[9px] text-muted">
                          {String(index + 1).padStart(2, "0")}
                        </span>

                        {column.name}
                      </label>

                      <span className="shrink-0 font-mono text-[9px] text-muted">
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
                        py-2
                        font-mono
                        text-xs
                        text-ink
                        placeholder:text-muted/50
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

            {manualMissing.length > 0 && (
              <div className="mt-3 rounded-panel border border-rule bg-ground px-3 py-2.5">
                <p className="text-xs font-medium text-ink">
                  Complete the required fields
                </p>

                <p className="mt-1 text-[10px] text-muted">
                  Missing:{" "}
                  <span className="font-mono text-ink">
                    {manualMissing.join(", ")}
                  </span>
                </p>
              </div>
            )}

            <div className="mt-3 flex justify-end">
              <button
                type="button"
                onClick={handlePredictManual}
                disabled={loading}
                className={`
                  rounded-panel
                  border
                  px-4
                  py-2
                  text-xs
                  font-medium
                  transition-all
                  ${
                    !loading
                      ? "cursor-pointer border-signal bg-signal text-white hover:opacity-90"
                      : "cursor-not-allowed border-rule bg-ground text-muted opacity-60"
                  }
                `}
              >
                {loading ? "Predicting…" : "Predict →"}
              </button>
            </div>
          </section>
        )}

        {/* Error */}
        {error && (
          <div className="rounded-panel border border-rule bg-ground px-3 py-2.5">
            <p className="text-xs font-medium text-ink">
              Prediction failed
            </p>
            <p className="mt-1 text-[10px] text-muted">
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
    <section className="rounded-panel border border-rule bg-surface p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-wider text-muted">
            Prediction complete
          </p>

          <h2 className="mt-0.5 text-sm font-semibold text-ink">
            Prediction results
          </h2>
        </div>

        <span className="rounded-full bg-ground px-2.5 py-1 font-mono text-[10px] text-ink">
          {result.n_samples}{" "}
          {result.n_samples === 1
            ? "prediction"
            : "predictions"}
        </span>
      </div>

      <div className="overflow-x-auto rounded-panel border border-rule">
        <table className="w-full min-w-[500px] text-left text-xs">
          <thead>
            <tr className="border-b border-rule bg-ground">
              <th className="px-3 py-2 text-[9px] font-medium uppercase tracking-wide text-muted">
                Row
              </th>

              <th className="px-3 py-2 text-[9px] font-medium uppercase tracking-wide text-muted">
                Prediction
              </th>

              {result.probabilities && (
                <th className="px-3 py-2 text-[9px] font-medium uppercase tracking-wide text-muted">
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
                <td className="px-3 py-2 font-mono text-[10px] text-muted">
                  {String(i).padStart(2, "0")}
                </td>

                <td className="px-3 py-2 font-mono text-xs text-ink">
                  {formatPrediction(prediction)}
                </td>

                {result.probabilities && (
                  <td className="px-3 py-2 font-mono text-[10px] text-ink">
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
        <p className="mt-2 font-mono text-[10px] text-muted">
          Showing first {MAX_ROWS_SHOWN} of {result.n_samples}.
        </p>
      )}
    </section>
  );
}