import { useState } from "react";
import { Dropzone } from "../Dropzone";
import { ScreenPanel } from "../ScreenPanel";
import { SplitSlider } from "../SplitSlider";
import { useDataset, type DataProfileResponse } from "../../hooks/useDataset";

const NO_TARGET = "__none__";

interface UploadScreenProps {
  profile: DataProfileResponse | null;
  onProfile: (profile: DataProfileResponse) => void;
  testSize: number;
  onTestSizeChange: (testSize: number) => void;
}

/**
 * Screen 1:
 * - Upload a CSV or choose a sample dataset
 * - Select the target column
 * - Configure the train/test split
 * - Show a compact dataset summary once loaded
 */
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

  // Kept so changing the target column can re-ingest the same file.
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
    <ScreenPanel maxWidthClassName="max-w-[720px]">
      <div className="flex flex-col gap-7">
        {/* Header */}
        <div>
          <p className="font-mono text-xs uppercase tracking-wider text-muted">
            Step 1 · Dataset
          </p>

          <h1 className="mt-1 text-xl font-semibold text-ink">
            Upload
          </h1>

          <p className="mt-1 text-sm leading-relaxed text-muted">
            Upload a CSV dataset or start with one of the sample datasets
            below.
          </p>
        </div>

        {/* File upload */}
        <section>
          <div className="mb-3">
            <h2 className="text-sm font-semibold text-ink">
              Dataset file
            </h2>

            <p className="mt-1 text-xs text-muted">
              CSV files with at least 50 rows and at most 100 columns.
            </p>
          </div>

          <Dropzone
            onFile={handleFile}
            loading={loading}
            fileName={file?.name ?? profile?.source}
          />

          {error && (
            <div className="mt-3 rounded-panel border border-rule bg-ground px-4 py-3">
              <p className="text-sm font-medium text-ink">
                Upload failed
              </p>

              <p className="mt-1 text-xs leading-relaxed text-muted">
                {error.message}
              </p>
            </div>
          )}
        </section>

        {/* Loaded dataset */}
        {profile && (
          <section className="rounded-panel border border-rule bg-ground p-5">
            <div className="mb-4 flex items-start justify-between gap-4">
              <div>
                <h2 className="text-sm font-semibold text-ink">
                  Dataset loaded
                </h2>

                <p className="mt-1 text-xs text-muted">
                  Review the dataset configuration before continuing.
                </p>
              </div>

              <span className="shrink-0 rounded-full bg-surface px-3 py-1.5 font-mono text-[10px] uppercase tracking-wide text-signal">
                ✓ Ready
              </span>
            </div>

            {/* Dataset summary */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              <DatasetStat
                label="Rows"
                value={profile.n_rows.toLocaleString()}
              />

              <DatasetStat
                label="Columns"
                value={profile.n_columns.toString()}
              />

              <DatasetStat
                label="Type"
                value={profile.data_type}
              />
            </div>

            {/* Target */}
            {file && (
              <div className="mt-5 border-t border-rule pt-5">
                <label
                  className="text-sm font-medium text-ink"
                  htmlFor="target-column"
                >
                  Target column
                </label>

                <p className="mt-1 text-xs text-muted">
                  Choose the column the model should predict, or select
                  no target for clustering.
                </p>

                <select
                  id="target-column"
                  className="
                    mt-3
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
                    hover:border-ink/40
                    focus:border-signal
                  "
                  value={profile.target_column ?? NO_TARGET}
                  disabled={loading}
                  onChange={(e) =>
                    handleTargetChange(e.target.value)
                  }
                >
                  {profile.columns.map((column) => (
                    <option key={column.name} value={column.name}>
                      {column.name}
                    </option>
                  ))}

                  <option value={NO_TARGET}>
                    No target (clustering)
                  </option>
                </select>
              </div>
            )}

            {/* Split */}
            <div className="mt-5 border-t border-rule pt-5">
              <h3 className="text-sm font-medium text-ink">
                Train / test split
              </h3>

              <p className="mt-1 mb-3 text-xs text-muted">
                Reserve part of the dataset for model evaluation.
              </p>

              <div className="rounded-panel border border-rule bg-surface p-4">
                <SplitSlider
                  testSize={testSize}
                  onChange={onTestSizeChange}
                  disabled={loading}
                />
              </div>
            </div>
          </section>
        )}

        {/* Sample datasets */}
        <section>
          <div className="mb-3">
            <h2 className="text-sm font-semibold text-ink">
              Sample datasets
            </h2>

            <p className="mt-1 text-xs text-muted">
              Try the platform without uploading your own data.
            </p>
          </div>

          {samplesError && (
            <div className="mb-3 rounded-panel border border-rule bg-ground px-4 py-3">
              <p className="text-xs text-muted">
                {samplesError.message}
              </p>
            </div>
          )}

          <div className="grid gap-3">
            {(samples ?? []).map((sample) => (
              <button
                key={sample.id}
                type="button"
                onClick={() => handleSample(sample.id)}
                disabled={loading}
                className="
                  group
                  w-full
                  cursor-pointer
                  rounded-panel
                  border
                  border-rule
                  bg-surface
                  p-4
                  text-left
                  transition-all
                  duration-150
                  hover:border-ink/40
                  hover:bg-ground
                  active:scale-[0.995]
                  disabled:cursor-not-allowed
                  disabled:opacity-50
                "
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-sm font-semibold text-ink">
                        {sample.name}
                      </span>

                      <span className="font-mono text-[10px] uppercase tracking-wide text-muted">
                        {sample.data_type}
                      </span>
                    </div>

                    <p className="mt-1 text-xs leading-relaxed text-muted">
                      {sample.description}
                    </p>
                  </div>

                  <span className="shrink-0 font-mono text-xs text-muted transition-colors group-hover:text-signal">
                    {sample.n_rows} × {sample.n_columns}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </section>
      </div>
    </ScreenPanel>
  );
}

/* -------------------------------------------------------------------------- */
/* Small dataset statistic                                                    */
/* -------------------------------------------------------------------------- */

function DatasetStat({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-panel border border-rule bg-surface px-3 py-3">
      <p className="font-mono text-[10px] uppercase tracking-wide text-muted">
        {label}
      </p>

      <p className="mt-1 truncate font-mono text-sm text-ink">
        {value}
      </p>
    </div>
  );
}