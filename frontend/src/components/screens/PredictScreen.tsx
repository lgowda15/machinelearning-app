import { useState } from "react";
import { Dropzone } from "../Dropzone";
import { ScreenPanel } from "../ScreenPanel";
import { buildSingleRowCsv, diffColumns, parseCsvHeader, type ColumnMismatch } from "../../lib/columns";
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

/** Screen 6 (frontend.md): two input modes, a toggle between them. Upload
 * CSV mirrors screen 1 and names a column mismatch client-side, ahead of
 * the backend's own rejection at Stage 6 (defense in depth, not a
 * replacement for it). Enter values renders one field per feature column
 * (confirmed available via `profile.columns` -- BUILD_SESSIONS.md Session
 * 8's pre-flight check) and submits by assembling a single-row CSV in the
 * browser, posting it through this same upload-based predict endpoint --
 * no new backend endpoint. */
export function PredictScreen({ profile, trainingResults }: PredictScreenProps) {
  const { result, loading, error, predict, reset } = usePrediction();
  const [modelKey, setModelKey] = useState(trainingResults?.results[0]?.model_key ?? "");
  const [mode, setMode] = useState<PredictMode>("csv");

  const [file, setFile] = useState<File | null>(null);
  const [mismatch, setMismatch] = useState<ColumnMismatch | null>(null);

  const [manualValues, setManualValues] = useState<Record<string, string>>({});
  const [manualMissing, setManualMissing] = useState<string[]>([]);

  if (!trainingResults || trainingResults.results.length === 0) {
    return (
      <ScreenPanel>
        <p className="text-sm text-muted">Train at least one model before predicting on new data.</p>
      </ScreenPanel>
    );
  }

  const featureColumns: ColumnSummary[] = profile.columns.filter(
    (column) => column.name !== profile.target_column,
  );
  const expectedColumns = featureColumns.map((column) => column.name);

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

  const handleManualFieldChange = (name: string, value: string) => {
    setManualValues((prev) => ({ ...prev, [name]: value }));
    setManualMissing((prev) => prev.filter((n) => n !== name));
  };

  const handlePredictManual = () => {
    const missing = featureColumns
      .filter((column) => !(manualValues[column.name] ?? "").trim())
      .map((column) => column.name);
    if (missing.length > 0) {
      setManualMissing(missing);
      return;
    }
    setManualMissing([]);
    const csv = buildSingleRowCsv(expectedColumns, manualValues);
    const manualFile = new File([csv], "manual-entry.csv", { type: "text/csv" });
    predict(trainingResults.training_id, modelKey, manualFile);
  };

  return (
    <ScreenPanel>
      <div className="flex flex-col gap-6">
        <div>
          <h1 className="text-lg font-medium text-ink">Predict on new data</h1>
          <p className="mt-1 text-sm text-muted">
            Run a trained model on new data -- upload a CSV with the same feature columns as
            training, or enter one row of values directly.
          </p>
        </div>

        <div className="rounded-panel border border-rule p-4">
          <label className="text-sm text-muted" htmlFor="predict-model">
            Model
          </label>
          <select
            id="predict-model"
            className="mt-1 w-full rounded-panel border border-rule bg-surface px-3 py-2 text-sm text-ink"
            value={modelKey}
            onChange={(e) => {
              setModelKey(e.target.value);
              reset();
            }}
          >
            {trainingResults.results.map((r) => (
              <option key={r.model_key} value={r.model_key}>
                {r.model_name}
              </option>
            ))}
          </select>
        </div>

        <div className="flex gap-2">
          {(["csv", "manual"] as const).map((m) => (
            <button
              key={m}
              type="button"
              aria-pressed={mode === m}
              onClick={() => switchMode(m)}
              className={
                "rounded-panel border px-4 py-2 text-sm " +
                (mode === m ? "border-signal text-signal" : "border-rule text-ink hover:border-ink")
              }
            >
              {m === "csv" ? "Upload CSV" : "Enter values"}
            </button>
          ))}
        </div>

        {mode === "csv" ? (
          <>
            <Dropzone onFile={handleFile} loading={loading} fileName={file?.name} />

            {mismatch && (
              <div className="rounded-panel border border-rule p-4 text-sm text-ink">
                <p className="font-medium">This file's columns don't match the training data.</p>
                {mismatch.missing.length > 0 && (
                  <p className="mt-1 text-xs text-muted">
                    Missing: <span className="font-mono text-ink">{mismatch.missing.join(", ")}</span>
                  </p>
                )}
                {mismatch.unexpected.length > 0 && (
                  <p className="mt-1 text-xs text-muted">
                    Unexpected: <span className="font-mono text-ink">{mismatch.unexpected.join(", ")}</span>
                  </p>
                )}
              </div>
            )}

            <button
              type="button"
              onClick={handlePredictCsv}
              disabled={!file || !!mismatch || loading}
              className="rounded-panel border border-signal bg-signal px-4 py-2 text-sm font-medium text-surface disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? "Predicting…" : "Predict"}
            </button>
          </>
        ) : (
          <>
            <div className="rounded-panel border border-rule p-4">
              <div className="flex flex-col gap-3">
                {featureColumns.map((column) => (
                  <div key={column.name} className="flex flex-col gap-1">
                    <label htmlFor={`manual-${column.name}`} className="text-sm text-muted">
                      {column.name}{" "}
                      <span className="font-mono text-xs lowercase text-muted">({column.dtype})</span>
                    </label>
                    <input
                      id={`manual-${column.name}`}
                      type={column.dtype === "numeric" ? "number" : "text"}
                      step={column.dtype === "numeric" ? "any" : undefined}
                      value={manualValues[column.name] ?? ""}
                      disabled={loading}
                      onChange={(e) => handleManualFieldChange(column.name, e.target.value)}
                      className={
                        "rounded-panel border bg-surface px-3 py-2 text-sm text-ink " +
                        (manualMissing.includes(column.name) ? "border-signal" : "border-rule")
                      }
                    />
                  </div>
                ))}
              </div>
            </div>

            {manualMissing.length > 0 && (
              <div className="rounded-panel border border-rule p-4 text-sm text-ink">
                <p className="font-medium">Enter a value for every feature before predicting.</p>
                <p className="mt-1 text-xs text-muted">
                  Missing: <span className="font-mono text-ink">{manualMissing.join(", ")}</span>
                </p>
              </div>
            )}

            <button
              type="button"
              onClick={handlePredictManual}
              disabled={loading}
              className="rounded-panel border border-signal bg-signal px-4 py-2 text-sm font-medium text-surface disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? "Predicting…" : "Predict"}
            </button>
          </>
        )}

        {error && <p className="text-sm text-ink">Prediction failed: {error.message}</p>}

        {result && <PredictionTable result={result} />}
      </div>
    </ScreenPanel>
  );
}

function formatPrediction(value: unknown): string {
  if (Array.isArray(value)) return value.map((v) => (typeof v === "number" ? v.toFixed(4) : String(v))).join(", ");
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  return String(value);
}

function PredictionTable({ result }: { result: components["schemas"]["PredictionResponse"] }) {
  const shown = result.predictions.slice(0, MAX_ROWS_SHOWN);

  return (
    <div>
      <p className="mb-2 text-sm font-medium uppercase tracking-wide text-muted">
        {result.n_samples} prediction{result.n_samples === 1 ? "" : "s"}
      </p>
      <div className="overflow-x-auto rounded-panel border border-rule bg-surface">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-rule text-xs uppercase tracking-wide text-muted">
              <th className="px-3 py-2 font-medium">Row</th>
              <th className="px-3 py-2 font-medium">Prediction</th>
              {result.probabilities && <th className="px-3 py-2 font-medium">Probabilities</th>}
            </tr>
          </thead>
          <tbody>
            {shown.map((prediction, i) => (
              <tr key={i} className="border-b border-rule last:border-b-0">
                <td className="px-3 py-2 font-mono text-xs text-muted">{i}</td>
                <td className="px-3 py-2 font-mono text-ink">{formatPrediction(prediction)}</td>
                {result.probabilities && (
                  <td className="px-3 py-2 font-mono text-xs text-ink">
                    {result.probabilities[i].map((p) => p.toFixed(3)).join(", ")}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {result.n_samples > MAX_ROWS_SHOWN && (
        <p className="mt-1 font-mono text-xs text-muted">
          Showing first {MAX_ROWS_SHOWN} of {result.n_samples}.
        </p>
      )}
    </div>
  );
}
