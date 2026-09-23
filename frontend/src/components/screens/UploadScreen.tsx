import { useState } from "react";
import { Dropzone } from "../Dropzone";
import { ScreenPanel } from "../ScreenPanel";
import { SplitSlider } from "../SplitSlider";
import {
  useDataset,
  type DataProfileResponse,
} from "../../hooks/useDataset";

const NO_TARGET = "__none__";

interface UploadScreenProps {
  profile: DataProfileResponse | null;
  onProfile: (profile: DataProfileResponse) => void;
  testSize: number;
  onTestSizeChange: (testSize: number) => void;
}

export function UploadScreen({
  profile,
  onProfile,
  testSize,
  onTestSizeChange,
}: UploadScreenProps) {
  const {
    samples,
    samplesError,
    loading,
    error,
    upload,
    loadSample,
  } = useDataset();

  const [file, setFile] = useState<File | null>(null);

  const handleFile = async (nextFile: File) => {
    setFile(nextFile);

    const result = await upload(nextFile, null, true);

    if (result) {
      onProfile(result);
    }
  };

  const handleTargetChange = async (value: string) => {
    if (!file) return;

    const result =
      value === NO_TARGET
        ? await upload(file, null, false)
        : await upload(file, value, true);

    if (result) {
      onProfile(result);
    }
  };

  const handleSample = async (sampleId: string) => {
    setFile(null);

    const result = await loadSample(sampleId);

    if (result) {
      onProfile(result);
    }
  };

  return (
    <ScreenPanel maxWidthClassName="max-w-[1400px]">
      <div className="space-y-2">
        {/* HEADER */}
        <div>
          <p className="font-mono text-[9px] uppercase tracking-[0.18em] text-muted">
            Step 1 · Dataset
          </p>

          <h1 className="mt-0.5 text-xl font-semibold tracking-tight text-ink">
            Upload
          </h1>

          <p className="text-xs text-muted">
            Upload a CSV dataset or start with one of the sample datasets.
          </p>
        </div>

        {/* TWO COLUMN LAYOUT */}
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          {/* ========================================================= */}
          {/* LEFT — SAMPLE DATASETS                                   */}
          {/* ========================================================= */}

          <div className="min-w-0">
            <section>
              <div className="mb-1.5">
                <h2 className="text-sm font-semibold text-ink">
                  Sample datasets
                </h2>

                <p className="text-[11px] text-muted">
                  Try the platform without uploading your own data.
                </p>
              </div>

              {samplesError && (
                <div className="mb-1.5 rounded-panel border border-rule bg-ground px-3 py-2">
                  <p className="text-[10px] text-muted">
                    {samplesError.message}
                  </p>
                </div>
              )}

              <div className="space-y-1.5">
                {(samples ?? []).map((sample) => (
                  <button
                    key={sample.id}
                    type="button"
                    onClick={() => handleSample(sample.id)}
                    disabled={loading}
                    className="
                      group
                      flex
                      w-full
                      items-center
                      justify-between
                      gap-3
                      rounded-panel
                      border
                      border-rule
                      bg-surface
                      px-3
                      py-2
                      text-left
                      transition-all
                      duration-150
                      hover:border-ink/40
                      hover:bg-ground
                      disabled:cursor-not-allowed
                      disabled:opacity-50
                    "
                  >
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-semibold text-ink">
                          {sample.name}
                        </span>

                        <span className="font-mono text-[8px] uppercase tracking-wide text-muted">
                          {sample.data_type}
                        </span>
                      </div>

                      <p className="truncate text-[10px] text-muted">
                        {sample.description}
                      </p>
                    </div>

                    <span className="shrink-0 font-mono text-[9px] text-muted">
                      {sample.n_rows} × {sample.n_columns}
                    </span>
                  </button>
                ))}
              </div>
            </section>

            {/* ======================================================= */}
            {/* DATASET LOADED — COMPACT                               */}
            {/* ======================================================= */}

            {profile && (
              <section className="mt-2 rounded-panel border border-rule bg-ground p-2.5">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <h2 className="text-xs font-semibold text-ink">
                      Dataset loaded
                    </h2>

                    <span className="rounded-full bg-surface px-2 py-0.5 font-mono text-[8px] uppercase text-signal">
                      ✓ Ready
                    </span>
                  </div>

                  <span className="font-mono text-[9px] text-muted">
                    {profile.data_type}
                  </span>
                </div>

                {/* Compact stats */}
                <div className="mt-2 grid grid-cols-3 gap-1.5">
                  <DatasetStat
                    label="Rows"
                    value={profile.n_rows.toLocaleString()}
                  />

                  <DatasetStat
                    label="Columns"
                    value={profile.n_columns.toString()}
                  />

                  <DatasetStat
                    label="Target"
                    value={profile.target_column ?? "None"}
                  />
                </div>

                {/* Target selector */}
                {file && (
                  <div className="mt-2 flex items-center gap-2">
                    <label
                      htmlFor="target-column"
                      className="shrink-0 text-[10px] font-medium text-ink"
                    >
                      Target
                    </label>

                    <select
                      id="target-column"
                      className="
                        min-w-0
                        flex-1
                        cursor-pointer
                        rounded-panel
                        border
                        border-rule
                        bg-surface
                        px-2
                        py-1.5
                        text-[11px]
                        text-ink
                        focus:border-signal
                        focus:outline-none
                      "
                      value={profile.target_column ?? NO_TARGET}
                      disabled={loading}
                      onChange={(e) =>
                        handleTargetChange(e.target.value)
                      }
                    >
                      {profile.columns.map((column) => (
                        <option
                          key={column.name}
                          value={column.name}
                        >
                          {column.name}
                        </option>
                      ))}

                      <option value={NO_TARGET}>
                        No target (clustering)
                      </option>
                    </select>
                  </div>
                )}
              </section>
            )}
          </div>

          {/* ========================================================= */}
          {/* RIGHT — FILE + SPLIT                                    */}
          {/* ========================================================= */}

          <div className="min-w-0">
            {/* DATASET FILE */}
            <section>
              <div className="mb-1.5">
                <h2 className="text-sm font-semibold text-ink">
                  Dataset file
                </h2>

                <p className="text-[11px] text-muted">
                  CSV files with at least 50 rows and at most 100 columns.
                </p>
              </div>

              <Dropzone
                onFile={handleFile}
                loading={loading}
                fileName={file?.name ?? profile?.source}
              />

              {error && (
                <div className="mt-1.5 rounded-panel border border-rule bg-ground px-3 py-2">
                  <p className="text-[10px] font-semibold text-ink">
                    Upload failed
                  </p>

                  <p className="text-[10px] text-muted">
                    {error.message}
                  </p>
                </div>
              )}
            </section>

            {/* TRAIN / TEST SPLIT */}
            <section className="mt-2 rounded-panel border border-rule bg-ground p-2.5">
              <div className="flex items-center justify-between gap-2">
                <div>
                  <h2 className="text-sm font-semibold text-ink">
                    Train / test split
                  </h2>

                  <p className="text-[10px] text-muted">
                    Reserve part of the dataset for evaluation.
                  </p>
                </div>

                <span className="shrink-0 rounded-full bg-surface px-2 py-1 font-mono text-[8px] uppercase text-muted">
                  Evaluation
                </span>
              </div>

              <div className="mt-2 rounded-panel border border-rule bg-surface px-3 py-2">
                <SplitSlider
                  testSize={testSize}
                  onChange={onTestSizeChange}
                  disabled={loading}
                />
              </div>
            </section>
          </div>
        </div>
      </div>
    </ScreenPanel>
  );
}

function DatasetStat({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="min-w-0 rounded-panel border border-rule bg-surface px-2 py-1.5">
      <p className="font-mono text-[7px] uppercase tracking-wide text-muted">
        {label}
      </p>

      <p className="mt-0.5 truncate font-mono text-[11px] text-ink">
        {value}
      </p>
    </div>
  );
}